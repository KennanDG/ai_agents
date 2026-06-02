from __future__ import annotations

from pathlib import Path
import difflib
import re
import ast
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from ai_agents.agents.coding.tools.filesystem import DEFAULT_IGNORES, read_file
from ai_agents.agents.coding.utils.search import is_ignored_context_path
from ai_agents.agents.coding.utils.constants import TEXT_SUFFIXES, PYTHON_SYMBOL_KINDS
from ai_agents.agents.coding.schemas import RepoSearchRequest, PythonSymbol, SearchMode




@dataclass(frozen=True)
class SearchResult:
    """A scored result from the repository search service."""

    path: str
    line_no: int | None
    snippet: str
    score: float
    reason: str
    query: str

    def to_line(self) -> str:
        line = self.line_no if self.line_no is not None else 1
        return f"{self.path}:{line}: {self.snippet} [score={self.score:.1f}; {self.reason}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line_no": self.line_no,
            "snippet": self.snippet,
            "score": self.score,
            "reason": self.reason,
            "query": self.query,
        }





class RepoSearchService:
    """Path-aware, ranked repository search.

    This service intentionally stays deterministic. LLM nodes can ask for structured
    searches, but the actual filtering, matching, scoring, and symbol indexing happen
    here.
    """

    def __init__(self, repo_root: Path, *, max_file_chars: int = 50_000) -> None:
        self.repo_root = repo_root.resolve()
        self.max_file_chars = max_file_chars
        self._symbol_index: list[PythonSymbol] | None = None


    def search(
        self,
        requests: Sequence[RepoSearchRequest | Mapping[str, Any] | str],
        *,
        max_results: int = 25,
    ) -> list[SearchResult]:
        
        normalized_requests = [self._normalize_request(request) for request in requests]
        results: list[SearchResult] = []

        for request in normalized_requests:
            request_results = self._search_one(request, max_results=request.max_results or max_results)
            results.extend(request_results)

        return self._dedupe_and_rank(results, max_results=max_results)
    


    def build_python_symbol_index(self) -> list[PythonSymbol]:
        if self._symbol_index is not None:
            return self._symbol_index

        symbols: list[PythonSymbol] = []

        for path in self._iter_candidate_files(
            RepoSearchRequest(file_extensions=[".py"], mode="symbol")
        ):
            rel = path.relative_to(self.repo_root).as_posix()

            try:
                text = read_file(self.repo_root, rel, max_chars=self.max_file_chars)
                tree = ast.parse(text)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    symbols.append(PythonSymbol(rel, node.name, "class", node.lineno))

                elif isinstance(node, ast.AsyncFunctionDef):
                    symbols.append(PythonSymbol(rel, node.name, "async_function", node.lineno))

                elif isinstance(node, ast.FunctionDef):
                    symbols.append(PythonSymbol(rel, node.name, "function", node.lineno))

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".", 1)[0]
                        symbols.append(PythonSymbol(rel, name, "import", node.lineno))

                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            symbols.append(PythonSymbol(rel, target.id, "constant", node.lineno))

        self._symbol_index = symbols
        return symbols



    def _search_one(self, request: RepoSearchRequest, *, max_results: int) -> list[SearchResult]:
        symbol_results = self._search_symbols(request)
        file_results = self._search_files(request)
        return self._dedupe_and_rank([*symbol_results, *file_results], max_results=max_results)



    def _search_symbols(self, request: RepoSearchRequest) -> list[SearchResult]:
        if request.file_extensions and ".py" not in request.file_extensions:
            return []

        terms = [term.lower() for term in request.terms if term]
        if not terms:
            return []

        results: list[SearchResult] = []

        for symbol in self.build_python_symbol_index():
            if not self._path_allowed(Path(symbol.path), request):
                continue

            symbol_name = symbol.name.lower()
            if request.mode == "symbol":
                matched = any(term == symbol_name or term in symbol_name for term in terms)
            elif request.mode == "all":
                matched = all(term in symbol_name or term in symbol.path.lower() for term in terms)
            elif request.mode == "exact":
                matched = any(term == symbol_name for term in terms)
            else:
                matched = any(term in symbol_name or term in symbol.path.lower() for term in terms)

            if not matched:
                continue

            score = self._base_path_score(symbol.path, request) + 12.0
            if any(term == symbol_name for term in terms):
                score += 8.0
            if symbol.kind in PYTHON_SYMBOL_KINDS:
                score += 2.0

            results.append(
                SearchResult(
                    path=symbol.path,
                    line_no=symbol.line_no,
                    snippet=f"{symbol.kind} {symbol.name}",
                    score=score,
                    reason=f"python symbol match: {symbol.kind} {symbol.name}",
                    query=self._query_label(request),
                )
            )

        return results



    def _search_files(self, request: RepoSearchRequest) -> list[SearchResult]:
        terms = [term.lower() for term in request.terms if term]
        results: list[SearchResult] = []

        for path in self._iter_candidate_files(request):
            rel = path.relative_to(self.repo_root).as_posix()
            rel_lower = rel.lower()

            try:
                text = read_file(self.repo_root, rel, max_chars=self.max_file_chars)
            except Exception:
                continue

            text_lower = text.lower()
            path_score = self._base_path_score(rel, request)
            matched_terms = self._matched_terms(terms, rel_lower, text_lower, request.mode)

            if terms and not matched_terms:
                continue

            # A path-filter-only request is useful for directory discovery.
            if not terms and not (request.path_includes or request.file_extensions):
                continue

            line_no, snippet = self._best_snippet(text, matched_terms or terms)
            score = path_score + self._content_score(rel_lower, text_lower, matched_terms, request)
            reason = self._reason(rel, request, matched_terms)

            results.append(
                SearchResult(
                    path=rel,
                    line_no=line_no,
                    snippet=snippet,
                    score=score,
                    reason=reason,
                    query=self._query_label(request),
                )
            )

        return results



    def _iter_candidate_files(self, request: RepoSearchRequest) -> Iterable[Path]:
        for path in sorted(self.repo_root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in DEFAULT_IGNORES for part in path.parts):
                continue
            if any(part.startswith(".") for part in path.relative_to(self.repo_root).parts):
                continue
            if not self._text_file_allowed(path):
                continue

            rel = path.relative_to(self.repo_root)
            rel_text = rel.as_posix()

            if is_ignored_context_path(rel_text):
                continue
            if not self._path_allowed(rel, request):
                continue

            yield path

    def _path_allowed(self, path: Path, request: RepoSearchRequest) -> bool:
        rel = path.as_posix().lower()

        if request.path_includes:
            includes = [item.lower().strip("/") for item in request.path_includes if item]
            if includes and not any(include in rel for include in includes):
                return False

        excludes = [item.lower().strip("/") for item in request.path_excludes if item]
        if excludes and any(exclude in rel for exclude in excludes):
            return False

        extensions = self._normalize_extensions(request.file_extensions)
        if extensions and self._suffix(path) not in extensions:
            return False

        return True


    def _text_file_allowed(self, path: Path) -> bool:
        return self._suffix(path) in TEXT_SUFFIXES


    def _suffix(self, path: Path) -> str:
        if path.name.lower() == "dockerfile":
            return ".dockerfile"
        return path.suffix.lower()


    def _normalize_request(self, request: RepoSearchRequest | Mapping[str, Any] | str) -> RepoSearchRequest:
        if isinstance(request, RepoSearchRequest):
            return request

        if isinstance(request, str):
            return RepoSearchRequest(terms=_tokenize_query(request), mode="all")

        terms = request.get("terms") or []
        if isinstance(terms, str):
            terms = _tokenize_query(terms)

        mode = request.get("mode") or "all"
        if mode not in {"all", "any", "exact", "symbol"}:
            mode = "all"

        max_results = request.get("max_results")
        if max_results is not None:
            try:
                max_results = max(1, int(max_results))
            except (TypeError, ValueError):
                max_results = None

        return RepoSearchRequest(
            terms=[str(term).strip() for term in terms if str(term).strip()],
            path_includes=_as_string_list(request.get("path_includes")),
            path_excludes=_as_string_list(request.get("path_excludes")),
            file_extensions=self._normalize_extensions(_as_string_list(request.get("file_extensions"))),
            mode=mode,  # type: ignore[arg-type]
            max_results=max_results,
        )


    def _normalize_extensions(self, extensions: Sequence[str] | None) -> list[str]:
        normalized: list[str] = []

        for extension in extensions or []:
            value = extension.strip().lower()
            if not value:
                continue
            if not value.startswith("."):
                value = f".{value}"
            normalized.append(value)

        return normalized


    def _matched_terms(
        self,
        terms: list[str],
        rel_lower: str,
        text_lower: str,
        mode: SearchMode,
    ) -> list[str]:
        if not terms:
            return []

        if mode == "exact":
            phrase = " ".join(terms)
            return [phrase] if phrase in text_lower or phrase in rel_lower else []

        matched = [term for term in terms if term in text_lower or term in rel_lower]

        if mode in {"all", "symbol"} and len(matched) != len(terms):
            return []

        return matched


    def _base_path_score(self, rel: str, request: RepoSearchRequest) -> float:
        rel_lower = rel.lower()
        filename = Path(rel_lower).name
        score = 0.0

        for include in request.path_includes:
            include_lower = include.lower().strip("/")
            if include_lower and include_lower in rel_lower:
                score += 10.0

        for term in request.terms:
            term_lower = term.lower()
            if term_lower in filename:
                score += 8.0
            elif term_lower in rel_lower:
                score += 4.0

        if any(part in rel_lower for part in ("test", "tests")):
            score += 1.5
        if len(rel_lower) > 120:
            score -= 2.0

        return score


    def _content_score(
        self,
        rel_lower: str,
        text_lower: str,
        matched_terms: list[str],
        request: RepoSearchRequest,
    ) -> float:
        score = 0.0

        if request.path_includes and any(
            include.lower().strip("/") in rel_lower for include in request.path_includes
        ):
            score += 10.0

        if not request.terms:
            return score + 2.0

        if len(matched_terms) == len(request.terms):
            score += 6.0
        else:
            score += len(matched_terms) * 2.0

        for term in matched_terms:
            if re.search(rf"\b(class|def|async def)\s+{re.escape(term)}\b", text_lower):
                score += 8.0
            elif re.search(rf"\b{re.escape(term)}\b", text_lower):
                score += 2.0

        return score


    def _best_snippet(self, text: str, terms: list[str]) -> tuple[int | None, str]:
        lines = text.splitlines()
        lower_terms = [term.lower() for term in terms if term]

        for line_no, line in enumerate(lines, start=1):
            line_lower = line.lower()
            if any(term in line_lower for term in lower_terms):
                return line_no, line.strip()[:300]

        for line_no, line in enumerate(lines, start=1):
            if line.strip():
                return line_no, line.strip()[:300]

        return None, "[empty file]"


    def _reason(self, rel: str, request: RepoSearchRequest, matched_terms: list[str]) -> str:
        parts: list[str] = []

        if request.path_includes:
            parts.append("path filter")
        if request.file_extensions:
            parts.append("extension filter")
        if matched_terms:
            parts.append("matched terms: " + ", ".join(matched_terms))
        if not parts:
            parts.append("repository text match")

        return "; ".join(parts)


    def _query_label(self, request: RepoSearchRequest) -> str:
        pieces: list[str] = []
        if request.terms:
            pieces.append("terms=" + ",".join(request.terms))
        if request.path_includes:
            pieces.append("path=" + ",".join(request.path_includes))
        if request.file_extensions:
            pieces.append("ext=" + ",".join(request.file_extensions))
        pieces.append(f"mode={request.mode}")
        return " | ".join(pieces)


    def _dedupe_and_rank(self, results: list[SearchResult], *, max_results: int) -> list[SearchResult]:
        best_by_path: dict[str, SearchResult] = {}

        for result in results:
            existing = best_by_path.get(result.path)
            if existing is None or result.score > existing.score:
                best_by_path[result.path] = result

        return sorted(
            best_by_path.values(),
            key=lambda result: (-result.score, result.path, result.line_no or 0),
        )[:max_results]





def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    
    if isinstance(value, str):
        return [value]
    
    return [str(item) for item in value if str(item).strip()]



def _tokenize_query(query: str) -> list[str]:
    return [token for token in re.split(r"\s+", query.strip()) if token]


def search_repository(
    repo_root: Path,
    requests: Sequence[RepoSearchRequest | Mapping[str, Any] | str],
    *,
    max_results: int = 25,
) -> list[SearchResult]:
    
    return RepoSearchService(repo_root).search(requests, max_results=max_results)



def format_search_results(results: Sequence[SearchResult]) -> list[str]:
    return [result.to_line() for result in results]



def search_repo(repo_root: Path, query: str, max_results: int = 25) -> list[str]:
    """Compatibility wrapper for simple repository search."""

    results = search_repository(
        repo_root,
        [RepoSearchRequest(terms=[query], mode="exact")],
        max_results=max_results,
    )

    return format_search_results(results)


def robust_search(repo_root: Path, queries: list[str] | str, max_results: int = 25) -> list[str]:
    """Compatibility wrapper around the structured search service.

    Prefer `search_repository()` for new code so callers can use path filters,
    extension filters, and match modes explicitly.
    """

    query_list = queries if isinstance(queries, list) else [queries]
    requests = [RepoSearchRequest(terms=_tokenize_query(query), mode="all") for query in query_list]
    results = search_repository(repo_root, requests, max_results=max_results)

    return format_search_results(results)


