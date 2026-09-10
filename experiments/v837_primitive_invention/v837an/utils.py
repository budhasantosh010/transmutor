from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
START_SHA = "f556c92a895b141f73f214e5eda3ac29b1927ea9"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_bytes(rel: str, rev: str = "HEAD") -> bytes:
    return subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=ROOT)


def git_blob_sha256(rel: str, rev: str = "HEAD") -> str:
    return hashlib.sha256(git_blob_bytes(rel, rev)).hexdigest()


def head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def now_stamps() -> tuple[str, str]:
    utc = datetime.now(timezone.utc)
    return utc.isoformat(), utc.astimezone(ZoneInfo("Asia/Dubai")).isoformat()


def deterministic_seed(*parts: object) -> int:
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def range_list(pair: list[int] | tuple[int, int]) -> list[int]:
    return list(range(int(pair[0]), int(pair[1]) + 1))
