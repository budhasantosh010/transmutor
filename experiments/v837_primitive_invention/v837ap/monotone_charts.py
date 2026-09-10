from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression

def _direction(h,z):
    c=float(np.corrcoef(np.asarray(h),np.asarray(z))[0,1]) if np.std(h)>1e-12 and np.std(z)>1e-12 else 1.0
    return 1 if c>=0 else -1

def _isotonic_values(h,z,direction):
    ir=IsotonicRegression(increasing=direction>0,out_of_bounds="clip");return ir.fit_transform(np.asarray(h,dtype=np.float64),np.asarray(z,dtype=np.float64))
def fit_monotone(h,z,kind:str,knots:int)->dict:
    h=np.asarray(h,dtype=np.float64).reshape(-1);z=np.asarray(z,dtype=np.float64).reshape(-1);direction=_direction(h,z);order=np.argsort(h,kind="mergesort");hs=h[order];zs=z[order];iso=_isotonic_values(hs,zs,direction)
    qs=np.linspace(0,1,knots);kh=np.quantile(hs,qs);kz=[]
    for x in kh:
        j=int(np.argmin(np.abs(hs-x)));kz.append(float(iso[j]))
    # enforce strict x uniqueness deterministically
    uh=[];uz=[]
    for x,y in zip(kh,kz):
        if uh and abs(float(x)-uh[-1])<1e-12: uz[-1]=float(y)
        else: uh.append(float(x));uz.append(float(y))
    if len(uh)<2:raise ValueError("degenerate monotone chart")
    chart={"kind":kind,"k":1,"knots_h":uh,"knots_z":uz,"direction":direction,"parameter_count":len(uh)*2,"fit_h_min":[float(np.min(h))],"fit_h_max":[float(np.max(h))],"fit_z_min":float(min(uz)),"fit_z_max":float(max(uz)),"gradient_steps":0}
    if kind=="MONOTONE_PCHIP_6":
        p=PchipInterpolator(np.asarray(uh),np.asarray(uz),extrapolate=False);grid=np.linspace(uh[0],uh[-1],512);der=p.derivative()(grid);consistent=bool(np.all(der>=-1e-10) if direction>0 else np.all(der<=1e-10));chart["derivative_sign_consistent"]=consistent
    return chart
def predict_monotone(chart,h):
    x=np.asarray(h,dtype=np.float64).reshape(-1);kh=np.asarray(chart["knots_h"]);kz=np.asarray(chart["knots_z"])
    if chart["kind"]=="MONOTONE_PCHIP_6":
        p=PchipInterpolator(kh,kz,extrapolate=False);cl=np.clip(x,kh[0],kh[-1]);return np.asarray(p(cl),dtype=np.float64)
    return np.interp(x,kh,kz)
def gradient_monotone(chart,h):
    x=float(np.asarray(h).reshape(-1)[0]);kh=np.asarray(chart["knots_h"]);kz=np.asarray(chart["knots_z"]);x=float(np.clip(x,kh[0],kh[-1]))
    if chart["kind"]=="MONOTONE_PCHIP_6":return np.asarray([float(PchipInterpolator(kh,kz,extrapolate=False).derivative()(x))])
    j=min(max(int(np.searchsorted(kh,x)-1),0),len(kh)-2);return np.asarray([(kz[j+1]-kz[j])/max(kh[j+1]-kh[j],1e-12)])
def inverse_monotone(chart,target,current_h=None):
    kh=np.asarray(chart["knots_h"]);kz=np.asarray(chart["knots_z"]);t=float(target)
    if t<min(kz)-1e-12 or t>max(kz)+1e-12:return None
    if chart["kind"]=="MONOTONE_PCHIP_6":
        lo,hi=float(kh[0]),float(kh[-1]);direction=int(chart["direction"])
        for _ in range(64):
            mid=(lo+hi)/2;v=float(predict_monotone(chart,[mid])[0])
            if (v<t)==(direction>0):lo=mid
            else:hi=mid
        return (lo+hi)/2
    if chart["direction"]<0:return float(np.interp(t,kz[::-1],kh[::-1]))
    return float(np.interp(t,kz,kh))
