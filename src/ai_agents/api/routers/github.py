from __future__ import annotations

import base64
import fnmatch
import json
import os

import shutil
import subprocess
import re
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from typing import Any, Literal
from fastapi import HTTPException, APIRouter, Query
from pathlib import Path

from ai_agents.api.schemas import (
    GitHubRepositoryImportRequest,
    GitHubRepositorySummary,
    GitHubRepositoryImportResponse,
    GitHubBranchSummary,
    GitHubConnectionTestResponse,
    GitHubRepositoryStatus,
    GitHubCreateBranchRequest,
    GitHubCreateBranchResponse,
    GitHubPullRequest,
    GitHubPullResponse,
    GitHubCommitRequest,
    GitHubCommitResponse, 
    GitHubPushRequest,
    GitHubPushResponse,
    GitHubPullRequestCreateRequest,
    GitHubPullRequestResponse,
)

from ai_agents.config.settings import settings



_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_IMPORT_LOCK = threading.RLock()
_AUTO_STASH_PREFIX = "ai-agents:auto-stash:"


class _AutoStashRestoreError(RuntimeError):
    def __init__(self, branch: str, stash_ref: str, detail: str) -> None:
        super().__init__(detail)
        self.branch = branch
        self.stash_ref = stash_ref
        self.detail = detail




router = APIRouter(prefix="/github", tags=["github"])



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
        method: str = "GET",
        query: dict[str, str | int] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        token = self._require_token()
        url = f"{self.api_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"

        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": settings.github_api_version,
            "User-Agent": "ai-agents",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"

        request = Request(url, data=payload, method=method, headers=headers)

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except HTTPError as exc:
            detail = exc.reason
            try:
                error_payload = json.loads(exc.read().decode("utf-8"))
                detail = error_payload.get("message") or detail
                errors = error_payload.get("errors")
                if errors:
                    detail = f"{detail}: {errors}"
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
    def _validate_branch(branch: str) -> str:
        normalized = branch.strip()
        invalid_character = any(char in normalized for char in " ~^:?*[\\\x7f")
        invalid = (
            not normalized
            or len(normalized) > 240
            or normalized.startswith(("-", ".", "/"))
            or normalized.endswith((".", "/", ".lock"))
            or ".." in normalized
            or "@{" in normalized
            or "//" in normalized
            or invalid_character
            or any(ord(char) < 32 for char in normalized)
        )
        if invalid:
            raise HTTPException(status_code=400, detail="Invalid Git branch name.")
        return normalized

    @staticmethod
    def _split_null_output(value: str) -> list[str]:
        return sorted({item for item in value.split("\0") if item})

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

    def _git_error_text(self, completed: subprocess.CompletedProcess[str]) -> str:
        detail = (completed.stderr or completed.stdout or "Git command failed.").strip()
        token = self.token or ""
        if token:
            detail = detail.replace(token, "***")
        return detail[-2_000:]

    def _raise_git_failure(
        self,
        completed: subprocess.CompletedProcess[str],
        *,
        operation: str,
    ) -> None:
        detail = self._git_error_text(completed)
        lowered = detail.casefold()

        auth_markers = (
            "authentication failed",
            "could not read username",
            "write access to repository not granted",
            "requested url returned error: 401",
            "requested url returned error: 403",
            "repository not found",
        )
        conflict_markers = (
            "not possible to fast-forward",
            "non-fast-forward",
            "would be overwritten",
            "conflict",
        )

        if any(marker in lowered for marker in auth_markers):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"GitHub Git transport rejected the {operation} operation. "
                    "Grant the configured credential access to this repository and at least "
                    "Contents: Read permission (Contents: Read and write for commits/pushes). "
                    "For a classic PAT, use the repo scope; for organization repositories, "
                    "also authorize SSO when required. Git reported: "
                    f"{detail}"
                ),
            )

        if any(marker in lowered for marker in conflict_markers):
            raise HTTPException(status_code=409, detail=detail)

        raise HTTPException(status_code=502, detail=detail)

    def _run_git(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if not shutil.which("git"):
            raise HTTPException(
                status_code=503,
                detail="Git is not installed on the ai_agents backend host.",
            )

        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(cwd) if cwd else None,
                env=self._git_env(),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Git operation timed out.") from exc

        if check and completed.returncode != 0:
            self._raise_git_failure(completed, operation=args[0] if args else "Git")
        return completed

    def _assert_git_transport_access(self, full_name: str, clone_url: str) -> None:
        completed = self._run_git(["ls-remote", "--heads", clone_url], check=False)
        if completed.returncode == 0:
            return

        detail = self._git_error_text(completed)
        lowered = detail.casefold()
        auth_markers = (
            "authentication failed",
            "could not read username",
            "write access to repository not granted",
            "requested url returned error: 401",
            "requested url returned error: 403",
            "repository not found",
        )
        if any(marker in lowered for marker in auth_markers):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"The GitHub API can see {full_name}, but the configured credential cannot "
                    "read its Git contents over HTTPS. Grant this credential repository access "
                    "and Contents: Read permission. For pushes and pull requests, grant Contents "
                    "and Pull requests: Read and write. For a classic PAT, use the repo scope; "
                    "authorize organization SSO when required. Git reported: "
                    f"{detail}"
                ),
            )

        raise HTTPException(
            status_code=502,
            detail=f"Could not verify Git transport access to {full_name}: {detail}",
        )

    def _repository_payload(self, full_name: str) -> dict[str, Any]:
        owner, name = self._validate_repository_name(full_name)
        payload = self._request_json(f"/repos/{owner}/{name}")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="GitHub returned an invalid repository response.")
        return payload

    def _checkout_path(self, full_name: str, *, require_exists: bool = True) -> Path:
        owner, name = self._validate_repository_name(full_name)
        target = (self.workspace_root / owner / name).resolve()
        if self.workspace_root not in target.parents:
            raise HTTPException(status_code=400, detail="Unsafe GitHub workspace path.")
        if require_exists and not (target / ".git").is_dir():
            raise HTTPException(
                status_code=409,
                detail="Repository has not been imported into the managed GitHub workspace.",
            )
        return target

    @staticmethod
    def _remote_repository_name(remote_url: str) -> str | None:
        normalized = remote_url.strip().rstrip("/")
        if normalized.endswith(".git"):
            normalized = normalized[:-4]

        if normalized.startswith("git@") and ":" in normalized:
            path = normalized.split(":", 1)[1]
        else:
            path = urlparse(normalized).path.lstrip("/")

        parts = [part for part in path.split("/") if part]
        return "/".join(parts[-2:]) if len(parts) >= 2 else None

    def _assert_origin_matches(self, repo_root: Path, full_name: str) -> None:
        origin = self._run_git(["remote", "get-url", "origin"], cwd=repo_root).stdout.strip()
        actual = self._remote_repository_name(origin)
        if not actual or actual.casefold() != full_name.casefold():
            raise HTTPException(
                status_code=409,
                detail=f"Managed checkout origin does not match {full_name}.",
            )

    def _configure_origin_fetch_all_branches(self, repo_root: Path) -> None:
        self._run_git(
            [
                "config",
                "--replace-all",
                "remote.origin.fetch",
                "+refs/heads/*:refs/remotes/origin/*",
            ],
            cwd=repo_root,
        )

    def _current_branch(self, repo_root: Path) -> str:
        completed = self._run_git(
            ["symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=repo_root,
            check=False,
        )
        branch = completed.stdout.strip()
        if completed.returncode != 0 or not branch:
            raise HTTPException(
                status_code=409,
                detail="The managed checkout is in detached HEAD state.",
            )
        return branch

    def _changed_files(self, repo_root: Path) -> tuple[list[str], list[str], list[str]]:
        staged = self._split_null_output(
            self._run_git(["diff", "--cached", "--name-only", "-z"], cwd=repo_root).stdout
        )
        unstaged = self._split_null_output(
            self._run_git(["diff", "--name-only", "-z"], cwd=repo_root).stdout
        )
        untracked = self._split_null_output(
            self._run_git(["ls-files", "--others", "--exclude-standard", "-z"], cwd=repo_root).stdout
        )
        return staged, unstaged, untracked

    @staticmethod
    def _auto_stash_message(branch: str) -> str:
        return f"{_AUTO_STASH_PREFIX}{branch}"

    def _find_branch_snapshot(self, repo_root: Path, branch: str) -> str | None:
        message = self._auto_stash_message(branch)
        result = self._run_git(
            ["stash", "list", "--format=%gd%x09%gs"],
            cwd=repo_root,
            check=False,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            stash_ref, separator, subject = line.partition("\t")
            if separator and subject.endswith(message):
                return stash_ref.strip() or None
        return None

    def _save_branch_snapshot(self, repo_root: Path, branch: str) -> bool:
        staged, unstaged, untracked = self._changed_files(repo_root)
        if not (staged or unstaged or untracked):
            return False

        result = self._run_git(
            [
                "stash",
                "push",
                "--include-untracked",
                "--message",
                self._auto_stash_message(branch),
            ],
            cwd=repo_root,
            check=False,
        )
        if result.returncode != 0:
            self._raise_git_failure(result, operation="stash")

        if not self._find_branch_snapshot(repo_root, branch):
            raise HTTPException(
                status_code=502,
                detail=f"Git did not create a recoverable branch snapshot for '{branch}'.",
            )
        return True

    def _restore_branch_snapshot(self, repo_root: Path, branch: str) -> bool:
        stash_ref = self._find_branch_snapshot(repo_root, branch)
        if not stash_ref:
            return False

        result = self._run_git(
            ["stash", "apply", "--index", stash_ref],
            cwd=repo_root,
            check=False,
        )
        if result.returncode != 0:
            raise _AutoStashRestoreError(
                branch=branch,
                stash_ref=stash_ref,
                detail=self._git_error_text(result),
            )

        self._run_git(["stash", "drop", stash_ref], cwd=repo_root)
        return True

    def _discard_worktree_changes(self, repo_root: Path) -> None:
        self._run_git(["reset", "--hard", "HEAD"], cwd=repo_root, check=False)
        self._run_git(["clean", "-fd"], cwd=repo_root, check=False)

    def _default_branch(self, full_name: str) -> str:
        repository = self._repository_payload(full_name)
        return str(repository.get("default_branch") or "main")

    def _assert_push_permission(self, full_name: str) -> None:
        repository = self._repository_payload(full_name)
        permissions = repository.get("permissions") or {}
        if not bool(permissions.get("push", False)):
            raise HTTPException(
                status_code=403,
                detail="The configured GitHub credential does not have push permission for this repository.",
            )

    def _assert_safe_publish_branch(self, full_name: str, branch: str) -> None:
        default_branch = self._default_branch(full_name)
        if (
            branch == default_branch
            and not settings.github_allow_default_branch_push
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Direct commits and pushes to the default branch '{default_branch}' are disabled. "
                    "Create an agent/* branch and open a pull request instead."
                ),
            )

    def _validate_commit_paths(self, repo_root: Path, paths: list[str]) -> list[str]:
        max_files = settings.github_max_commit_files
        if len(paths) > max_files:
            raise HTTPException(
                status_code=400,
                detail=f"A single agent commit is limited to {max_files} files.",
            )

        staged, unstaged, untracked = self._changed_files(repo_root)
        changed = set(staged) | set(unstaged) | set(untracked)
        normalized_paths: list[str] = []

        for raw_path in paths:
            path = raw_path.replace("\\", "/").strip().lstrip("/")
            candidate = (repo_root / path).resolve()
            if not path or repo_root not in candidate.parents:
                raise HTTPException(status_code=400, detail=f"Unsafe commit path: {raw_path}")
            if path not in changed:
                raise HTTPException(
                    status_code=409,
                    detail=f"Requested commit path is not currently changed: {path}",
                )

            lowered = path.casefold()
            basename = Path(lowered).name
            for pattern in settings.github_blocked_path_patterns:
                normalized_pattern = pattern.casefold()
                if fnmatch.fnmatch(lowered, normalized_pattern) or fnmatch.fnmatch(basename, normalized_pattern):
                    raise HTTPException(
                        status_code=409,
                        detail=f"Blocked sensitive path cannot be committed by the agent: {path}",
                    )

            if candidate.exists() and candidate.is_file():
                size = candidate.stat().st_size
                if size > settings.github_max_file_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the agent commit size limit "
                            f"({settings.github_max_file_size_bytes} bytes): {path}"
                        ),
                    )
            normalized_paths.append(path)

        return sorted(set(normalized_paths))

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

    def test_connection(self, full_name: str | None) -> GitHubConnectionTestResponse:
        status = self.status()
        git_available = shutil.which("git") is not None
        workspace_writable = False
        git_transport_connected = False
        permissions: dict[str, bool] = {}
        default_branch: str | None = None
        normalized_full_name: str | None = None
        transport_error: str | None = None

        try:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
            workspace_writable = os.access(self.workspace_root, os.W_OK)
        except OSError:
            workspace_writable = False

        if full_name:
            owner, name = self._validate_repository_name(full_name)
            normalized_full_name = f"{owner}/{name}"
            repository = self._repository_payload(normalized_full_name)
            default_branch = str(repository.get("default_branch") or "main")
            repository_permissions = repository.get("permissions") or {}
            permissions = {
                "admin": bool(repository_permissions.get("admin", False)),
                "maintain": bool(repository_permissions.get("maintain", False)),
                "push": bool(repository_permissions.get("push", False)),
                "triage": bool(repository_permissions.get("triage", False)),
                "pull": bool(repository_permissions.get("pull", True)),
            }
            clone_url = str(repository.get("clone_url") or f"https://github.com/{owner}/{name}.git")
            if git_available:
                completed = self._run_git(["ls-remote", "--heads", clone_url], check=False)
                git_transport_connected = completed.returncode == 0
                if not git_transport_connected:
                    transport_error = self._git_error_text(completed)
        else:
            git_transport_connected = git_available

        connected = bool(
            status.get("connected")
            and git_available
            and workspace_writable
            and git_transport_connected
        )
        if connected:
            message = "GitHub API, Git transport, and managed workspace checks passed."
        elif normalized_full_name and transport_error:
            message = (
                f"The GitHub API can see {normalized_full_name}, but Git transport cannot read "
                "the repository. Grant the configured credential repository access and "
                "Contents: Read permission; authorize organization SSO when required. "
                f"Git reported: {transport_error}"
            )
        else:
            message = "One or more GitHub connection checks failed."

        return GitHubConnectionTestResponse(
            connected=connected,
            api_connected=bool(status.get("connected")),
            git_available=git_available,
            git_transport_connected=git_transport_connected,
            workspace_writable=workspace_writable,
            token_kind=self.token_kind,
            account=status.get("account"),
            full_name=normalized_full_name,
            default_branch=default_branch,
            permissions=permissions,
            message=message,
        )

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

    def import_repository(
        self,
        *,
        full_name: str,
        requested_ref: str | None,
        refresh: bool,
    ) -> GitHubRepositoryImportResponse:
        owner, name = self._validate_repository_name(full_name)
        repository = self._repository_payload(full_name)
        default_branch = str(repository.get("default_branch") or "main")
        ref = self._validate_branch(requested_ref or default_branch)
        clone_url = str(repository.get("clone_url") or f"https://github.com/{owner}/{name}.git")

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        target = self._checkout_path(full_name, require_exists=False)
        saved_previous_changes = False
        restored_target_changes = False
        previous_ref: str | None = None

        with _IMPORT_LOCK:
            reused = target.exists()

            if reused:
                if not (target / ".git").is_dir():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Managed workspace exists but is not a Git repository: {target}",
                    )
                self._assert_origin_matches(target, full_name)
                self._configure_origin_fetch_all_branches(target)
                previous_ref = self._current_branch(target)

                if previous_ref == ref:
                    if refresh:
                        staged, unstaged, untracked = self._changed_files(target)
                        if staged or unstaged or untracked:
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    f"Cannot refresh '{ref}' while it has local changes. "
                                    "Switching to another branch is allowed; the changes will be "
                                    "saved automatically for this branch."
                                ),
                            )
                        self._run_git(
                            [
                                "fetch",
                                "--prune",
                                "origin",
                                f"refs/heads/{ref}:refs/remotes/origin/{ref}",
                            ],
                            cwd=target,
                        )
                        self._run_git(["merge", "--ff-only", f"origin/{ref}"], cwd=target)
                else:
                    saved_previous_changes = self._save_branch_snapshot(target, previous_ref)

                    try:
                        self._run_git(
                            [
                                "fetch",
                                "--prune",
                                "origin",
                                f"refs/heads/{ref}:refs/remotes/origin/{ref}",
                            ],
                            cwd=target,
                        )
                        local_exists = self._run_git(
                            ["show-ref", "--verify", "--quiet", f"refs/heads/{ref}"],
                            cwd=target,
                            check=False,
                        ).returncode == 0
                        if local_exists:
                            self._run_git(["switch", ref], cwd=target)
                            self._run_git(["merge", "--ff-only", f"origin/{ref}"], cwd=target)
                        else:
                            self._run_git(
                                ["switch", "--create", ref, "--track", f"origin/{ref}"],
                                cwd=target,
                            )
                        restored_target_changes = self._restore_branch_snapshot(target, ref)
                    except Exception as exc:
                        self._discard_worktree_changes(target)
                        current_after_failure = self._current_branch(target)
                        if current_after_failure != previous_ref:
                            self._run_git(["switch", previous_ref], cwd=target)

                        rollback_error: str | None = None
                        if saved_previous_changes:
                            try:
                                self._restore_branch_snapshot(target, previous_ref)
                            except Exception as restore_exc:
                                rollback_error = str(restore_exc)

                        if isinstance(exc, _AutoStashRestoreError):
                            detail = (
                                f"Could not restore the saved changes for '{exc.branch}' after switching. "
                                f"The checkout was returned to '{previous_ref}', and {exc.stash_ref} "
                                f"was retained for manual recovery. Git reported: {exc.detail}"
                            )
                            if rollback_error:
                                detail += f" Previous-branch recovery also needs attention: {rollback_error}"
                            raise HTTPException(status_code=409, detail=detail) from exc

                        if rollback_error:
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    f"Branch switch failed and the previous branch snapshot could not "
                                    f"be restored automatically: {rollback_error}"
                                ),
                            ) from exc
                        raise
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self._assert_git_transport_access(full_name, clone_url)
                    self._run_git(
                        [
                            "clone",
                            "--depth",
                            "1",
                            "--branch",
                            ref,
                            "--no-single-branch",
                            clone_url,
                            str(target),
                        ]
                    )
                except Exception:
                    shutil.rmtree(target, ignore_errors=True)
                    raise

        actual_ref = self._current_branch(target)
        return GitHubRepositoryImportResponse(
            full_name=f"{owner}/{name}",
            ref=actual_ref,
            repo_root=str(target),
            reused_existing_checkout=reused,
            previous_ref=previous_ref,
            saved_previous_changes=saved_previous_changes,
            restored_target_changes=restored_target_changes,
        )

    def repository_status(self, full_name: str) -> GitHubRepositoryStatus:
        repo_root = self._checkout_path(full_name)
        self._assert_origin_matches(repo_root, full_name)
        branch = self._current_branch(repo_root)
        staged, unstaged, untracked = self._changed_files(repo_root)
        head_sha = self._run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
        upstream_result = self._run_git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            cwd=repo_root,
            check=False,
        )
        upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else None
        ahead = 0
        behind = 0
        if upstream:
            counts = self._run_git(
                ["rev-list", "--left-right", "--count", f"{upstream}...HEAD"],
                cwd=repo_root,
                check=False,
            )
            if counts.returncode == 0:
                pieces = counts.stdout.strip().split()
                if len(pieces) == 2:
                    behind, ahead = int(pieces[0]), int(pieces[1])

        return GitHubRepositoryStatus(
            full_name=full_name,
            repo_root=str(repo_root),
            branch=branch,
            default_branch=self._default_branch(full_name),
            head_sha=head_sha,
            upstream=upstream,
            ahead=ahead,
            behind=behind,
            dirty=bool(staged or unstaged or untracked),
            staged_files=staged,
            unstaged_files=unstaged,
            untracked_files=untracked,
        )

    def create_branch(
        self,
        *,
        full_name: str,
        branch: str,
        base: str | None,
    ) -> GitHubCreateBranchResponse:
        repo_root = self._checkout_path(full_name)
        self._assert_origin_matches(repo_root, full_name)
        self._assert_push_permission(full_name)
        branch = self._validate_branch(branch)
        current_branch = self._current_branch(repo_root)
        staged, unstaged, untracked = self._changed_files(repo_root)

        if branch == self._default_branch(full_name):
            raise HTTPException(status_code=409, detail="The agent branch cannot be the default branch.")

        with _IMPORT_LOCK:
            local_exists = self._run_git(
                ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                cwd=repo_root,
                check=False,
            ).returncode == 0
            if local_exists:
                raise HTTPException(status_code=409, detail=f"Local branch already exists: {branch}")

            remote_exists = self._run_git(
                ["ls-remote", "--exit-code", "--heads", "origin", branch],
                cwd=repo_root,
                check=False,
            ).returncode == 0
            if remote_exists:
                raise HTTPException(status_code=409, detail=f"Remote branch already exists: {branch}")

            start_point = "HEAD"
            if base and base != current_branch:
                if staged or unstaged or untracked:
                    raise HTTPException(
                        status_code=409,
                        detail="Commit or discard local changes before creating a branch from a different base.",
                    )
                base = self._validate_branch(base)
                self._run_git(
                    [
                        "fetch",
                        "origin",
                        f"refs/heads/{base}:refs/remotes/origin/{base}",
                    ],
                    cwd=repo_root,
                )
                start_point = f"origin/{base}"

            self._run_git(["switch", "--create", branch, start_point], cwd=repo_root)
            sha = self._run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()

        return GitHubCreateBranchResponse(full_name=full_name, branch=branch, sha=sha)

    def pull(self, full_name: str) -> GitHubPullResponse:
        repo_root = self._checkout_path(full_name)
        self._assert_origin_matches(repo_root, full_name)
        branch = self._current_branch(repo_root)
        staged, unstaged, untracked = self._changed_files(repo_root)
        if staged or unstaged or untracked:
            raise HTTPException(
                status_code=409,
                detail="Cannot pull while the managed checkout has uncommitted changes.",
            )

        before = self._run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
        with _IMPORT_LOCK:
            self._run_git(
                ["fetch", "--prune", "origin", f"refs/heads/{branch}:refs/remotes/origin/{branch}"],
                cwd=repo_root,
            )
            self._run_git(["merge", "--ff-only", f"origin/{branch}"], cwd=repo_root)
        after = self._run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
        return GitHubPullResponse(
            full_name=full_name,
            branch=branch,
            head_sha=after,
            changed=before != after,
        )

    def commit(self, request: GitHubCommitRequest) -> GitHubCommitResponse:
        full_name = request.full_name
        repo_root = self._checkout_path(full_name)
        self._assert_origin_matches(repo_root, full_name)
        self._assert_push_permission(full_name)
        branch = self._current_branch(repo_root)
        self._assert_safe_publish_branch(full_name, branch)
        message = request.message.strip()
        paths = self._validate_commit_paths(repo_root, request.paths)

        pre_staged, _, _ = self._changed_files(repo_root)
        unrelated_staged = sorted(set(pre_staged) - set(paths))
        if unrelated_staged:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Refusing to include previously staged files outside the requested scope: "
                    + ", ".join(unrelated_staged)
                ),
            )

        with _IMPORT_LOCK:
            self._run_git(["add", "--", *paths], cwd=repo_root)
            staged, _, _ = self._changed_files(repo_root)
            unrelated_after_stage = sorted(set(staged) - set(paths))
            if unrelated_after_stage:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Staging produced changes outside the requested scope: "
                        + ", ".join(unrelated_after_stage)
                    ),
                )
            if not staged:
                raise HTTPException(status_code=409, detail="No staged changes remain to commit.")

            self._run_git(
                [
                    "-c",
                    f"user.name={settings.github_commit_author_name}",
                    "-c",
                    f"user.email={settings.github_commit_author_email}",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "--no-gpg-sign",
                    "-m",
                    message,
                ],
                cwd=repo_root,
            )
            commit_sha = self._run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()

        return GitHubCommitResponse(
            full_name=full_name,
            branch=branch,
            commit_sha=commit_sha,
            committed_files=staged,
        )

    def push(self, full_name: str) -> GitHubPushResponse:
        repo_root = self._checkout_path(full_name)
        self._assert_origin_matches(repo_root, full_name)
        self._assert_push_permission(full_name)
        branch = self._current_branch(repo_root)
        self._assert_safe_publish_branch(full_name, branch)
        staged, unstaged, untracked = self._changed_files(repo_root)
        if staged or unstaged or untracked:
            raise HTTPException(
                status_code=409,
                detail="Commit all intended changes before pushing the branch.",
            )

        with _IMPORT_LOCK:
            fetch = self._run_git(
                ["fetch", "origin", f"refs/heads/{branch}:refs/remotes/origin/{branch}"],
                cwd=repo_root,
                check=False,
            )
            remote_exists = fetch.returncode == 0
            if remote_exists:
                counts = self._run_git(
                    ["rev-list", "--left-right", "--count", f"origin/{branch}...HEAD"],
                    cwd=repo_root,
                ).stdout.strip().split()
                behind = int(counts[0]) if len(counts) == 2 else 0
                ahead = int(counts[1]) if len(counts) == 2 else 0
                if behind > 0:
                    raise HTTPException(
                        status_code=409,
                        detail="Remote branch contains commits that are not in the managed checkout. Pull/rebase before pushing.",
                    )
                if ahead == 0:
                    head_sha = self._run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
                    return GitHubPushResponse(
                        full_name=full_name,
                        branch=branch,
                        commit_sha=head_sha,
                        pushed=False,
                    )

            self._run_git(
                [
                    "-c",
                    "core.hooksPath=/dev/null",
                    "push",
                    "--porcelain",
                    "--set-upstream",
                    "origin",
                    f"HEAD:refs/heads/{branch}",
                ],
                cwd=repo_root,
            )
            head_sha = self._run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()

        return GitHubPushResponse(
            full_name=full_name,
            branch=branch,
            commit_sha=head_sha,
            pushed=True,
        )

    def create_pull_request(
        self,
        request: GitHubPullRequestCreateRequest,
    ) -> GitHubPullRequestResponse:
        full_name = request.full_name
        owner, name = self._validate_repository_name(full_name)
        repo_root = self._checkout_path(full_name)
        self._assert_origin_matches(repo_root, full_name)
        current_branch = self._current_branch(repo_root)
        head = self._validate_branch(request.head or current_branch)
        base = self._validate_branch(request.base or self._default_branch(full_name))
        if head != current_branch:
            raise HTTPException(
                status_code=409,
                detail="Pull requests may only be opened from the currently checked out managed branch.",
            )
        if head == base:
            raise HTTPException(status_code=409, detail="Pull request head and base branches must differ.")
        staged, unstaged, untracked = self._changed_files(repo_root)
        if staged or unstaged or untracked:
            raise HTTPException(
                status_code=409,
                detail="Commit or discard local changes before creating a pull request.",
            )

        self._assert_push_permission(full_name)
        remote_head = self._run_git(
            ["ls-remote", "--exit-code", "--heads", "origin", head],
            cwd=repo_root,
            check=False,
        )
        if remote_head.returncode != 0:
            raise HTTPException(
                status_code=409,
                detail=f"Push branch '{head}' before creating a pull request.",
            )

        existing = self._request_json(
            f"/repos/{owner}/{name}/pulls",
            query={
                "state": "open",
                "head": f"{owner}:{head}",
                "base": base,
                "per_page": 1,
            },
        )
        if isinstance(existing, list) and existing:
            item = existing[0]
            return GitHubPullRequestResponse(
                full_name=full_name,
                number=int(item["number"]),
                title=str(item["title"]),
                html_url=str(item["html_url"]),
                base=str(item["base"]["ref"]),
                head=str(item["head"]["ref"]),
                draft=bool(item.get("draft", False)),
                created=False,
            )

        item = self._request_json(
            f"/repos/{owner}/{name}/pulls",
            method="POST",
            body={
                "title": request.title.strip(),
                "body": request.body,
                "head": head,
                "base": base,
                "draft": request.draft,
                "maintainer_can_modify": request.maintainer_can_modify,
            },
        )
        return GitHubPullRequestResponse(
            full_name=full_name,
            number=int(item["number"]),
            title=str(item["title"]),
            html_url=str(item["html_url"]),
            base=str(item["base"]["ref"]),
            head=str(item["head"]["ref"]),
            draft=bool(item.get("draft", False)),
            created=True,
        )




@router.get("/status")
def github_status() -> dict[str, Any]:
    return GitHubService().status()


@router.get("/connection-test", response_model=GitHubConnectionTestResponse)
def github_connection_test(
    full_name: str | None = Query(default=None, description="Optional owner/name repository to test."),
) -> GitHubConnectionTestResponse:
    return GitHubService().test_connection(full_name)


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


@router.get("/repositories/status", response_model=GitHubRepositoryStatus)
def github_repository_status(
    full_name: str = Query(..., description="Repository full name (owner/name)"),
) -> GitHubRepositoryStatus:
    return GitHubService().repository_status(full_name)


@router.post("/repositories/branches/create", response_model=GitHubCreateBranchResponse)
def github_create_branch(request: GitHubCreateBranchRequest) -> GitHubCreateBranchResponse:
    return GitHubService().create_branch(
        full_name=request.full_name,
        branch=request.branch,
        base=request.base,
    )


@router.post("/repositories/pull", response_model=GitHubPullResponse)
def github_pull(request: GitHubPullRequest) -> GitHubPullResponse:
    return GitHubService().pull(request.full_name)


@router.post("/repositories/commit", response_model=GitHubCommitResponse)
def github_commit(request: GitHubCommitRequest) -> GitHubCommitResponse:
    return GitHubService().commit(request)


@router.post("/repositories/push", response_model=GitHubPushResponse)
def github_push(request: GitHubPushRequest) -> GitHubPushResponse:
    return GitHubService().push(request.full_name)


@router.post("/repositories/pull-requests", response_model=GitHubPullRequestResponse)
def github_create_pull_request(
    request: GitHubPullRequestCreateRequest,
) -> GitHubPullRequestResponse:
    return GitHubService().create_pull_request(request)
