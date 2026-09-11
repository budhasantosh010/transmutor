from __future__ import annotations

import hashlib, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
START_SHA = "287d37743751554c68742bc275b558d7f9bb1f95"
REQUIRED_BRANCH = "research/v837-causal-operator-canonicalization-program-ir"
LOCAL_BRANCH_ALIAS = "research-v837-causal-operator-canonicalization-program-ir"
AQ = ROOT / "experiments/v837_primitive_invention/v837aq"


def read_json(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def _default(x):
    try:
        import numpy as np
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
    except Exception: pass
    raise TypeError(type(x).__name__)
def canonical_json(obj)->bytes: return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False,default=_default).encode()
def sha256_json(obj)->str: return hashlib.sha256(canonical_json(obj)).hexdigest()
def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def write_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=_default)+"\n",encoding="utf-8",newline="\n")
def head_sha()->str:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def branch_name()->str:return subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
def git_blob_sha256(rel:str,rev:str="HEAD")->str:return hashlib.sha256(subprocess.check_output(["git","show",f"{rev}:{rel}"],cwd=ROOT)).hexdigest()
def nrmse(y,p,scale=2.0)->float:
    import numpy as np
    y=np.asarray(y,float);p=np.asarray(p,float)
    return float(np.sqrt(np.mean((p-y)**2))/scale) if len(y) else float("inf")
