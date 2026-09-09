from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.v837_primitive_invention.v837ak.dynamic_fingerprint import generate_all_fingerprints, load_size_matrix
from experiments.v837_primitive_invention.v837ak.utils import HERE, normalize, robust_center_scale, write_json


def _distance(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.linalg.norm(a-b,axis=1)/math.sqrt(a.shape[1])


def calibrate_reliability()->dict:
    generate_all_fingerprints()
    sizes={}; thresholds={}
    for k in range(1,11):
        records,d1=load_size_matrix(k,"d1",competent_only=True); _,d2=load_size_matrix(k,"d2",competent_only=True)
        combined=np.concatenate([d1,d2],axis=0); median,iqr=robust_center_scale(combined); n1=normalize(d1,median,iqr); n2=normalize(d2,median,iqr)
        self_dist=_distance(n1,n2)
        per_org=math.comb(10,k); shifted=np.roll(n2,-per_org,axis=0); nonself=_distance(n1,shifted)
        labels=np.concatenate([np.ones(len(self_dist)),np.zeros(len(nonself))]); scores=-np.concatenate([self_dist,nonself])
        auc=float(roc_auc_score(labels,scores)); med_self=float(np.median(self_dist)); med_non=float(np.median(nonself)); p90=float(np.percentile(self_dist,90)); tau=float(np.percentile(self_dist,95))
        passed=auc>=0.80 and med_self<=0.60*med_non and p90<=med_non
        sizes[str(k)]={
            "size":k,"eligible":bool(passed),"self_vs_nonself_roc_auc":auc,"median_self_distance":med_self,"median_nonself_distance":med_non,
            "self_to_nonself_ratio":med_self/max(med_non,1e-12),"p90_self_distance":p90,"tau":tau,"occurrences":len(records),"fingerprint_dimension":int(d1.shape[1]),
            "failure_code":None if passed else f"DYNAMIC_FINGERPRINT_UNRELIABLE_SIZE_{k}",
            "self_distance_summary":{"p50":med_self,"p90":p90,"p95":tau,"max":float(np.max(self_dist))},
            "nonself_distance_summary":{"p50":med_non,"p10":float(np.percentile(nonself,10)),"p90":float(np.percentile(nonself,90))},
        }
        thresholds[str(k)]={"tau":tau,"median":median.tolist(),"iqr":iqr.tolist(),"dimension":int(d1.shape[1]),"eligible":bool(passed)}
        print(f"size {k}: AUC={auc:.4f} self/non={med_self/max(med_non,1e-12):.4f} p90={p90:.4f} nonself_med={med_non:.4f} eligible={passed}",flush=True)
    payload={"version":"V837ak","stage":"AK4","sizes":sizes,"eligible_sizes":[k for k in range(1,11) if sizes[str(k)]["eligible"]],"all_sizes_failed":not any(v["eligible"] for v in sizes.values())}
    write_json(HERE/"diagnostics/fingerprint_reliability.json",payload); write_json(HERE/"diagnostics/fingerprint_thresholds.json",{"version":"V837ak","sizes":thresholds})
    return payload


def main()->int:
    p=calibrate_reliability(); print(json.dumps({"eligible_sizes":p["eligible_sizes"],"all_sizes_failed":p["all_sizes_failed"]},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
