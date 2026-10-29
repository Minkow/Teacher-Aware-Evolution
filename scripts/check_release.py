"""Check release content without printing potential secret values."""
import ast
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}
PATTERNS = {
    "credential-like token": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "service token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})"),
    "bearer credential": re.compile(r"Bearer\s+[A-Za-z0-9_.-]{24,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "private absolute path": re.compile(r"/(?:Users|zhanglijun|data-ai)/[^\s\"']+"),
    "internal endpoint": re.compile(r"https?://(?:10\.|192\.168\.|100\.(?:64|66)\.)"),
}


def main():
    failures, count = [], 0
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_symlink():
            failures.append((str(rel), "symlink"))
            continue
        if not path.is_file():
            continue
        count += 1
        if any(part in {"trash", "temp", "tmp", ".ruff_cache", ".mypy_cache", "htmlcov"} for part in rel.parts) or path.name in {".DS_Store", ".coverage"}:
            failures.append((str(rel), "temporary artifact"))
        if any(part in {"programs", "seeds", "generated_programs", "runs", "results", "data", "checkpoints", ".secrets"} for part in rel.parts):
            failures.append((str(rel), "non-source artifact"))
        if path.suffix.lower() in {".pkl", ".pickle", ".pt", ".pth", ".npy", ".npz", ".zip", ".tgz", ".pem", ".key"}:
            failures.append((str(rel), "binary/data/credential artifact"))
        if path.name in {".env", "credentials.json", "token"} or ".secrets" in rel.parts:
            failures.append((str(rel), "credential file"))
        if path.stat().st_size > 5_000_000:
            failures.append((str(rel), "unexpected large artifact"))
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            failures.append((str(rel), "unexpected binary file"))
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append((str(rel), label))
        if path.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError:
                failures.append((str(rel), "invalid Python syntax"))
    for path, reason in failures:
        print(f"FAIL {path}: {reason}")
    print(f"Scanned {count} files; {len(failures)} findings")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
