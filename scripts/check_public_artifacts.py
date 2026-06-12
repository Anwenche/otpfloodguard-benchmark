#!/usr/bin/env python3
"""Check public text artifacts for local paths and obvious secret patterns."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".txt",
    ".yml",
    ".yaml",
}
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "audit",
    "build",
    "__MACOSX",
}
PATH_PATTERN = (
    r"(/" + r"Users/"
    + r"|/" + r"home/"
    + r"|C:" + r"\\\\Users\\\\"
    + r"|New " + r"project"
    + r"|Docu" + r"ments/)"
)
PATTERNS = {
    "absolute_path": re.compile(PATH_PATTERN),
    "private_key": re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "api_key_like": re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
}


def iter_public_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def main() -> int:
    findings: list[str] = []
    for path in iter_public_text_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{rel}: cannot decode as UTF-8")
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{rel}:{line_no}: {name}")

    if findings:
        print("Public artifact check failed:")
        for finding in findings:
            print(f"  - {finding}")
        return 1
    print(f"Public artifact check passed ({len(iter_public_text_files())} text files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
