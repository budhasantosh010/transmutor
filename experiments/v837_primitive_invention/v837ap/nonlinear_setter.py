from __future__ import annotations
import numpy as np
from .chart_reader import read_chart
from .gradient_writer import gradient_step
from .inverse_writer import set_1d_inverse,set_binary_prototype


def set_state_geometry(state,q,chart,target,d90:float,binary:bool,writer_family:str="AUTO",tangent=None,max_iterations:int=4):
    s=np.asarray(state,dtype=np.float64).reshape(40);Q=np.asarray(q,dtype=np.float64);Q=Q.reshape(40,-1);h=s@Q
    current=float(read_chart(chart,h.reshape(1,-1))[0]);rz=max(float(chart.get("fit_z_max",1))-float(chart.get("fit_z_min",-1)),1e-12)
    if abs(current-float(target))/rz<=1e-4:
        return {"valid":True,"state":s.copy(),"iterations":0,"gradient_norm":None,"delta_h":np.zeros_like(h).tolist(),"intervention_norm":0.0,"read_after":current}
    if Q.shape[1]==1 and binary and writer_family!="TANGENT_FIELD":return set_binary_prototype(s,Q,chart,target)
    if Q.shape[1]==1 and chart["kind"] in {"POLYNOMIAL","MONOTONE_PWL_4","MONOTONE_PWL_8","MONOTONE_PCHIP_6"} and writer_family!="TANGENT_FIELD":
        return set_1d_inverse(s,Q,chart,target)
    cur=h.copy();last_g=None;iters=0;fail=None
    for r in range(max_iterations):
        before=float(read_chart(chart,cur.reshape(1,-1))[0]);rz=max(float(chart.get("fit_z_max",1))-float(chart.get("fit_z_min",-1)),1e-12)
        if abs(before-float(target))/rz<=1e-4:break
        if writer_family=="TANGENT_FIELD" and tangent is not None:
            from .tangent_field import tangent_step
            step=tangent_step(chart,tangent,cur,target,d90);last_g=step.get("gradient_norm")
        else:
            step=gradient_step(chart,cur,target,d90);last_g=step.get("gradient_norm")
        if not step.get("valid"):fail=step.get("failure_code");break
        cur=np.asarray(step["h"],dtype=np.float64);iters=r+1
    if fail:return {"valid":False,"failure_code":fail,"state":s,"iterations":iters,"gradient_norm":last_g,"delta_h":None}
    ds=Q@(cur-h);return {"valid":True,"state":s+ds,"iterations":iters,"gradient_norm":last_g,"delta_h":np.asarray(cur-h).tolist(),"intervention_norm":float(np.linalg.norm(ds)),"read_after":float(read_chart(chart,cur.reshape(1,-1))[0])}
