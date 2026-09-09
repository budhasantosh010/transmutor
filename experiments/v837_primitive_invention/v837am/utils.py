from __future__ import annotations

import hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
START_SHA="7d3b70ee6907b5502e0f229116d74eff4745c71e"


def read_json(path:Path): return json.loads(path.read_text(encoding="utf-8"))
def canonical_json(obj)->bytes: return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
def sha256_json(obj)->str: return hashlib.sha256(canonical_json(obj)).hexdigest()
def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def git_blob_bytes(rel:str,rev:str="HEAD")->bytes: return subprocess.check_output(["git","show",f"{rev}:{rel}"],cwd=ROOT)
def git_blob_sha256(rel:str,rev:str="HEAD")->str: return hashlib.sha256(git_blob_bytes(rel,rev)).hexdigest()
def write_json(path:Path,obj)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
def seeds(lo:int,hi:int)->list[int]: return list(range(lo,hi+1))
def now_stamps()->tuple[str,str]:
    u=datetime.now(timezone.utc); return u.isoformat(),u.astimezone(ZoneInfo("Asia/Dubai")).isoformat()
def head_sha()->str: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
