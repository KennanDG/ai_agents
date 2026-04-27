from __future__ import annotations

from dataclasses import dataclass, field



@dataclass(frozen=True)
class ValidationCommand:
    command: str
    reason: str = ""



@dataclass
class ValidationResult:
    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "reason": self.reason,
            "passed": self.passed,
        }



@dataclass
class ValidationSuiteResult:
    profile: str
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def to_dicts(self) -> list[dict]:
        return [result.to_dict() for result in self.results]