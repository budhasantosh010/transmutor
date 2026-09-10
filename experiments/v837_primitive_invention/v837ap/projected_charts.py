from __future__ import annotations

from math import comb
import numpy as np
from .binary_charts import fit_logistic
from .polynomial_charts import fit_polynomial

PROJECTED_FAMILIES=("LINEAR","QUADRATIC","CUBIC")

def degree_for(kind:str)->int:return {"LINEAR":1,"QUADRATIC":2,"CUBIC":3}[kind]
def fit_projected_chart(h,z,kind:str,binary:bool):
    d=degree_for(kind);return fit_logistic(h,z,d) if binary else fit_polynomial(h,z,d)
def expected_feature_count(k:int,degree:int|str)->int:
    d=degree_for(degree) if isinstance(degree,str) else int(degree)
    return comb(int(k)+d,d)
