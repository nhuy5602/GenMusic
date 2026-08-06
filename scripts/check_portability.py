"""Fail when submission-visible files depend on local machines or notes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISALLOWED_PATH_PARTS = {
    "local_notes",
    "evidence",
}
DISALLOWED_NAMES = {
    "defense_guide.md",
    "report_conformance.md",
    "report_conformance.json",
}
CONTENT_PATTERNS = {
    "absolute Windows user path": re.compile(
        r"[A-Za-z]:\\Users\\[^\\\r\n]+",
        re.IGNORECASE,
    ),
    "embedded Kaggle API token": re.compile(r"KGAT_[A-Za-z0-9_-]{24,}"),
}


def git_visible_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=project_root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return [
        project_root / line
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def audit_portability(project_root: Path = PROJECT_ROOT) -> dict:
    findings: list[dict[str, str]] = []
    files = git_visible_files(project_root)
    for path in files:
        relative = path.relative_to(project_root)
        folded_parts = {part.casefold() for part in relative.parts}
        if folded_parts & DISALLOWED_PATH_PARTS:
            findings.append({
                "path": relative.as_posix(),
                "reason": "local/report-only directory is submission-visible",
            })
            continue
        if relative.name.casefold() in DISALLOWED_NAMES:
            findings.append({
                "path": relative.as_posix(),
                "reason": "report/defense-only file is submission-visible",
            })
            continue
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for reason, pattern in CONTENT_PATTERNS.items():
            if pattern.search(content):
                findings.append({
                    "path": relative.as_posix(),
                    "reason": reason,
                })
    return {
        "status": "passed" if not findings else "failed",
        "visible_files": len(files),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    report = audit_portability(args.root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
