from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FREEZE_PATH = (
    PROJECT_ROOT
    / "04_MANUSCRIPT"
    / "01_FRONTIERS_PSYCHOLOGY_WORKING"
    / "EVIDENCE_FREEZE.md"
)
ROW_PATTERN = re.compile(
    r"^\| `(?P<path>[^`]+)` \|.*\| `(?P<sha256>[0-9a-f]{64})` \|$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_frozen_manuscript_evidence() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for line in FREEZE_PATH.read_text(encoding="utf-8").splitlines():
        match = ROW_PATTERN.match(line)
        if not match:
            continue
        relative = Path(match.group("path"))
        expected = match.group("sha256")
        artifact = PROJECT_ROOT / relative
        actual = sha256_file(artifact) if artifact.is_file() else None
        rows.append(
            {
                "path": relative.as_posix(),
                "exists": artifact.is_file(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )

    passed = len(rows) == 12 and all(bool(row["match"]) for row in rows)
    return {
        "status": "PASS" if passed else "FAIL",
        "freeze_file": str(FREEZE_PATH.relative_to(PROJECT_ROOT)),
        "artifact_count": len(rows),
        "mismatch_count": sum(not bool(row["match"]) for row in rows),
        "artifacts": rows,
    }


def main() -> None:
    result = audit_frozen_manuscript_evidence()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
