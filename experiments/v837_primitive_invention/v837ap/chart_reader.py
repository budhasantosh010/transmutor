from __future__ import annotations
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import rankdata
from .polynomial_charts import predict_polynomial,gradient_polynomial,inverse_polynomial_1d
from .monotone_charts import predict_monotone,gradient_monotone,inverse_monotone
from .binary_charts import predict_logistic,gradient_logistic,predict_binary_isotonic

BINARY_FAMILIES={"conditional_routing","delayed_recall"}

def read_chart(chart,h):
    kind=chart["kind"]
    if kind=="POLYNOMIAL":return predict_polynomial(chart,h)
    if kind in {"MONOTONE_PWL_4","MONOTONE_PWL_8","MONOTONE_PCHIP_6"}:return predict_monotone(chart,h)
    if kind=="LOGISTIC":return predict_logistic(chart,h)
    if kind.startswith("ISOTONIC_"):return predict_binary_isotonic(chart,h)
    raise KeyError(kind)
def chart_gradient(chart,h):
    kind=chart["kind"]
    if kind=="POLYNOMIAL":return gradient_polynomial(chart,h)
    if kind in {"MONOTONE_PWL_4","MONOTONE_PWL_8","MONOTONE_PCHIP_6"}:return gradient_monotone(chart,h)
    if kind=="LOGISTIC":return gradient_logistic(chart,h)
    if kind.startswith("ISOTONIC_"):
        x=float(np.asarray(h).reshape(-1)[0]);kh=np.asarray(chart["knots_h"]);kp=np.asarray(chart["knots_p"]);j=min(max(int(np.searchsorted(kh,x)-1),0),len(kh)-2);return np.asarray([2*(kp[j+1]-kp[j])/max(kh[j+1]-kh[j],1e-12)])
    raise KeyError(kind)
def inverse_1d(chart,target,current_h):
    kind=chart["kind"]
    if kind=="POLYNOMIAL":return inverse_polynomial_1d(chart,target,current_h)
    if kind in {"MONOTONE_PWL_4","MONOTONE_PWL_8","MONOTONE_PCHIP_6"}:return inverse_monotone(chart,target,current_h)
    raise ValueError("no direct inverse")
def reader_metrics(pred,truth,rz:float,binary:bool)->dict:
    p=np.asarray(pred,dtype=np.float64).reshape(-1);z=np.asarray(truth,dtype=np.float64).reshape(-1);nrmse=float(np.sqrt(np.mean((p-z)**2))/max(rz,1e-12))
    pearson=float(np.corrcoef(z,p)[0,1]) if len(z)>1 and np.std(z)>1e-12 and np.std(p)>1e-12 else 0.;spearman=float(np.corrcoef(rankdata(z),rankdata(p))[0,1]) if len(z)>1 and np.std(z)>1e-12 and np.std(p)>1e-12 else 0.
    if binary:
        yt=(z>0).astype(int);yp=(p>0).astype(int);acc=float(np.mean(yt==yp));bal=float(balanced_accuracy_score(yt,yp));passed=nrmse<=.15 and acc>=.95 and bal>=.95
        return {"normalized_rmse":nrmse,"classification_accuracy":acc,"balanced_accuracy":bal,"pearson":pearson,"spearman":spearman,"n":len(z),"pass":bool(passed)}
    passed=nrmse<=.10 and abs(pearson)>=.95 and spearman>=.95
    return {"normalized_rmse":nrmse,"pearson":pearson,"spearman":spearman,"n":len(z),"pass":bool(passed)}
