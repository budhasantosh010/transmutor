from __future__ import annotations
from .polynomial_charts import fit_polynomial
from .monotone_charts import fit_monotone
from .binary_charts import fit_logistic,fit_binary_isotonic

CONTINUOUS_K1=("AFFINE","POLYNOMIAL_2","POLYNOMIAL_3","MONOTONE_PWL_4","MONOTONE_PWL_8","MONOTONE_PCHIP_6")
BINARY_K1=("AFFINE","LOGISTIC_MONOTONE","ISOTONIC_4","ISOTONIC_8")

def fit_scalar_chart(h,z,kind:str,binary:bool):
    if kind=="AFFINE":return fit_polynomial(h,z,1)
    if kind=="POLYNOMIAL_2":return fit_polynomial(h,z,2)
    if kind=="POLYNOMIAL_3":return fit_polynomial(h,z,3)
    if kind=="MONOTONE_PWL_4":return fit_monotone(h,z,"MONOTONE_PWL_4",4)
    if kind=="MONOTONE_PWL_8":return fit_monotone(h,z,"MONOTONE_PWL_8",8)
    if kind=="MONOTONE_PCHIP_6":return fit_monotone(h,z,"MONOTONE_PCHIP_6",6)
    if kind=="LOGISTIC_MONOTONE":return fit_logistic(h,z,1)
    if kind=="ISOTONIC_4":return fit_binary_isotonic(h,z,4)
    if kind=="ISOTONIC_8":return fit_binary_isotonic(h,z,8)
    raise KeyError(kind)
