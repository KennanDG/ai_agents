from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ai_agents.config.settings import settings


router = APIRouter(prefix="/github", tags=["github"])

_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_IMPORT_LOCK = threading.RLock()


class GitHubRepositoryImportRequest(BaseModel):
    full_name: str = Field(..., examples=["owner/repository"])
    ref: str | None = Field(default=None, description="Branch or tag to check out.")
    refresh: bool = Field(
        default=False,
        description="Fetch and hard-reset an existing managed checkout.",
    )


class GitHubRepositorySummary(BaseModel):
    id: int
    full_name: str
    name: str
    owner: str
    private: bool
    default_branch: str
    clone_url: str
    html_url: str
    updated_at: str | None = None
    permissions: dict[str, bool] = Field(default_factory=dict)


class GitHubRepositoryImportResponse(BaseModel):
    full_name: str
    ref: str
    repo_root: str
    reused_existing_checkout: bool


class GitHubBranchSummary(BaseModel):
    name: str
    sha: str


class GitHubService:
    def __init__(self) -> None:
        self.api_url = settings.github_api_url.rstrip("/")
        self.token = settings.resolved_github_token()
        self.token_kind: Literal["user", "installation"] = settings.github_token_kind
        self.workspace_root = Path(settings.github_workspace_root).expanduser().resolve()
        self.timeout_seconds = settings.github_timeout_seconds

    def _require_token(self) -> str:
        if not self.token:
            raise HTTPException(
                status_code=503,
                detail=(
                    "GitHub is not configured. Set GITHUB_TOKEN (or GITHUB_SECRET_ARN) "
                    "on the backend."
                ),
            )
        return self.token

    def _request_json(
        self,
        path: str,
        *,
        query: dict[str, str | int] | None = None,
    ) -> Any:
        token = self._require_token()
        url = f"{self.api_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"

        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": settings.github_api_version,
                "User-Agent": "ai-agents",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.reason
            try:
                payload = json.loads(exc.read().decode("utf-8"))
                detail = payload.get("message") or detail
            except Exception:
                pass
            raise HTTPException(
                status_code=exc.code,
                detail=f"GitHub API request failed: {detail}",
            ) from exc
        except URLError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Could not reach GitHub: {exc.reason}",
            ) from exc

    @staticmethod
    def _normalize_repository(item: dict[str, Any]) -> GitHubRepositorySummary:
        owner = item.get("owner") or {}
        permissions = item.get("permissions") or {}

        return GitHubRepositorySummary(
            id=int(item["id"]),
            full_name=str(item["full_name"]),
            name=str(item["name"]),
            owner=str(owner.get("login") or ""),
            private=bool(item.get("private", False)),
            default_branch=str(item.get("default_branch") or "main"),
            clone_url=str(item.get("clone_url") or ""),
            html_url=str(item.get("html_url") or ""),
            updated_at=item.get("updated_at"),
            permissions={
                "admin": bool(permissions.get("admin", False)),
                "maintain": bool(permissions.get("maintain", False)),
                "push": bool(permissions.get("push", False)),
                "triage": bool(permissions.get("triage", False)),
                "pull": bool(permissions.get("pull", True)),
            },
        )

    def status(self) -> dict[str, Any]:
        if not self.token:
            return {
                "connected": False,
                "token_kind": self.token_kind,
                "account": None,
            }

        account: str | None = None
        if self.token_kind == "user":
            user = self._request_json("/user")
            account = str(user.get("login") or "") or None
        else:
            self._request_json("/installation/repositories", query={"per_page": 1})
            account = "GitHub App installation"

        return {
            "connected": True,
            "token_kind": self.token_kind,
            "account": account,
        }

    def list_repositories(self, *, page: int, per_page: int) -> list[GitHubRepositorySummary]:
        if self.token_kind == "installation":
            payload = self._request_json(
                "/installation/repositories",
                query={"page": page, "per_page": per_page},
            )
            raw_repositories = payload.get("repositories", [])
        else:
            raw_repositories = self._request_json(
                "/user/repos",
                query={
                    "page": page,
                    "per_page": per_page,
                    "sort": "updated",
                    "direction": "desc",
                    "affiliation": "owner,collaborator,organization_member",
                },
            )

        return [
            self._normalize_repository(item)
            for item in raw_repositories
            if isinstance(item, dict)
        ]

    def list_branches(
        self,
        full_name: str,
        *,
        page: int = 1,
        per_page: int = 100,
    ) -> list[GitHubBranchSummary]:
        owner, name = self._validate_repository_name(full_name)
        raw_branches = self._request_json(
            f"/repos/{owner}/{name}/branches",
            query={"page": page, "per_page": per_page},
        )
        return [
            GitHubBranchSummary(
                name=str(item["name"]),
                sha=str(item["commit"]["sha"]),
            )
            for item in raw_branches
            if isinstance(item, dict)
        ]

    @staticmethod
    def _validate_repository_name(full_name: str) -> tuple[str, str]:
        normalized = full_name.strip()
        if not _REPOSITORY_RE.fullmatch(normalized):
            raise HTTPException(
                status_code=400,
                detail="Repository must use the owner/name format.",
            )
        owner, name = normalized.split("/", 1)
        return owner, name

    @staticmethod
    def _validate_ref(ref: str) -> str:
        normalized = ref.strip()
        if (
            not normalized
            or not _REF_RE.fullmatch(normalized)
            or normalized.startswith("/")
            or normalized.endswith("/")
            or ".." in normalized.split("/")
        ):
            raise HTTPException(status_code=400, detail="Invalid Git reference.")
        return normalized

    def _git_env(self) -> dict[str, str]:
        token = self._require_token()
        basic = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
        env = os.environ.copy()
        env.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
            }
        )
        return env

    def _run_git(self, args: list[str], *, cwd: Path | None = None) -> None:
        if not shutil.which("git"):
            raise HTTPException(
                status_code=503,
                detail="Git is not installed on the ai_agents backend host.",
            )

        try:
            subprocess.run(
                ["git", *args],
                cwd=str(cwd) if cwd else None,
                env=self._git_env(),
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Git operation timed out.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "Git command failed.").strip()
            raise HTTPException(status_code=502, detail=stderr[-2_000:]) from exc

    def import_repository(
        self,
        *,
        full_name: str,
        requested_ref: str | None,
        refresh: bool,
    ) -> GitHubRepositoryImportResponse:
        owner, name = self._validate_repository_name(full_name)
        repository = self._request_json(f"/repos/{owner}/{name}")
        default_branch = str(repository.get("default_branch") or "main")
        ref = self._validate_ref(requested_ref or default_branch)
        clone_url = str(repository.get("clone_url") or f"https://github.com/{owner}/{name}.git")

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        target = (self.workspace_root / owner / name).resolve()
        if self.workspace_root not in target.parents:
            raise HTTPException(status_code=400, detail="Unsafe GitHub workspace path.")

        with _IMPORT_LOCK:
            reused = target.exists()

            if reused:
                if not (target / ".git").is_dir():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Managed workspace exists but is not a Git repository: {target}",
                    )

                if refresh:
                    self._run_git(["fetch", "origin", ref, "--depth", "1"], cwd=target)
                    self._run_git(["checkout", "-B", ref, "FETCH_HEAD"], cwd=target)
                    self._run_git(["reset", "--hard", "FETCH_HEAD"], cwd=target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self._run_git(
                        [
                            "clone",
                            "--depth",
                            "1",
                            "--branch",
                            ref,
                            "--single-branch",
                            clone_url,
                            str(target),
                        ]
                    )
                except Exception:
                    shutil.rmtree(target, ignore_errors=True)
                    raise

        return GitHubRepositoryImportResponse(
            full_name=f"{owner}/{name}",
            ref=ref,
            repo_root=str(target),
            reused_existing_checkout=reused,
        )


@router.get("/status")
def github_status() -> dict[str, Any]:
    return GitHubService().status()


@router.get("/repositories", response_model=list[GitHubRepositorySummary])
def github_repositories(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=100, ge=1, le=100),
) -> list[GitHubRepositorySummary]:
    return GitHubService().list_repositories(page=page, per_page=per_page)


@router.post("/repositories/import", response_model=GitHubRepositoryImportResponse)
def import_github_repository(
    request: GitHubRepositoryImportRequest,
) -> GitHubRepositoryImportResponse:
    return GitHubService().import_repository(
        full_name=request.full_name,
        requested_ref=request.ref,
        refresh=request.refresh,
    )


@router.get("/repositories/branches", response_model=list[GitHubBranchSummary])
def github_branches(
    full_name: str = Query(..., description="Repository full name (owner/name)"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=100, ge=1, le=100),
) -> list[GitHubBranchSummary]:
    return GitHubService().list_branches(
        full_name=full_name,
        page=page,
        per_page=per_page,
    )
