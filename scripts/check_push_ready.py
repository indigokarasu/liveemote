#!/usr/bin/env python3
"""Validate that this branch is safe to push as a PR.

The demo intentionally downloads large third-party repositories into vendor/ at
setup time. Those working trees must stay ignored and untracked; otherwise a PR
push can become enormous or fail on hosted Git size limits.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_TRACKED_BYTES = 25 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {
    ".css", ".example", ".gitignore", ".gitkeep", ".html", ".js", ".json",
    ".md", ".py", ".toml", ".txt", ".yaml", ".yml", "",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def tracked_files() -> list[Path]:
    output = git("ls-files")
    return [ROOT / line for line in output.splitlines() if line]


def is_binary_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return False
    try:
        path.read_text(encoding="utf-8")
        return False
    except UnicodeDecodeError:
        return True


def main() -> int:
    files = tracked_files()
    nested_git_files = [p for p in files if ".git" in p.relative_to(ROOT).parts]
    if nested_git_files:
        fail("tracked nested .git metadata found: " + ", ".join(str(p.relative_to(ROOT)) for p in nested_git_files[:5]))

    tracked_vendor_payload = [p for p in files if p.parts[len(ROOT.parts):1 + len(ROOT.parts)] == ("vendor",) and p.name != "README.md"]
    if tracked_vendor_payload:
        fail("vendor payload is tracked; keep cloned upstream repos ignored: " + ", ".join(str(p.relative_to(ROOT)) for p in tracked_vendor_payload[:5]))

    binary_files = [p for p in files if p.is_file() and is_binary_file(p)]
    if binary_files:
        fail("tracked binary files are not supported in the GitHub PR diff: " + ", ".join(str(p.relative_to(ROOT)) for p in binary_files[:10]))

    large_files = [p for p in files if p.is_file() and p.stat().st_size > MAX_TRACKED_BYTES]
    if large_files:
        fail("tracked file exceeds 25 MiB push-safety limit: " + ", ".join(str(p.relative_to(ROOT)) for p in large_files))

    ignored_vendor_dirs = [p for p in (ROOT / "vendor").glob("*") if p.is_dir() and (p / ".git").exists()]
    if ignored_vendor_dirs:
        ignored = git("check-ignore", *[str(p.relative_to(ROOT)) for p in ignored_vendor_dirs]).splitlines()
        missing = sorted(set(str(p.relative_to(ROOT)) for p in ignored_vendor_dirs) - set(ignored))
        if missing:
            fail("cloned vendor repos are not ignored: " + ", ".join(missing))

    status = git("status", "--porcelain")
    if status:
        fail("working tree has uncommitted tracked changes:\n" + status)

    remote = git("remote")
    if not remote:
        print("WARNING: no git remote is configured in this local checkout; pushing requires a remote in the caller environment.")

    print("Push-ready: tracked payload is small, vendor clones are ignored, and tracked working tree is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
