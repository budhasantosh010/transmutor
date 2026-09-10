from __future__ import annotations
import numpy as np
from .chart_reader import inverse_1d,read_chart

def set_1d_inverse(state,q,chart,target):
    s=np.asarray(state,dtype=np.float64).reshape(40);q=np.asarray(q,dtype=np.float64).reshape(40,1);h=float((s@q)[0]);hz=inverse_1d(chart,float(target),h)
    if hz is None:return {"valid":False,"failure_code":"POLYNOMIAL_CHART_NO_VALID_INVERSE","state":s,"iterations":0,"gradient_norm":None,"delta_h":None}
    ds=q[:,0]*(float(hz)-h);patched=s+ds;return {"valid":True,"state":patched,"iterations":1,"gradient_norm":None,"delta_h":[float(hz-h)],"intervention_norm":float(np.linalg.norm(ds)),"read_after":float(read_chart(chart,np.array([[float(hz)]],dtype=np.float64))[0])}

def set_binary_prototype(state,q,chart,target):
    s=np.asarray(state,dtype=np.float64).reshape(40);q=np.asarray(q,dtype=np.float64).reshape(40,1);proto=chart.get("prototype_plus") if float(target)>0 else chart.get("prototype_minus")
    if proto is None:return {"valid":False,"failure_code":"BINARY_PROTOTYPE_MISSING","state":s,"iterations":0,"gradient_norm":None,"delta_h":None}
    h=float((s@q)[0]);ds=q[:,0]*(float(proto)-h);patched=s+ds;return {"valid":True,"state":patched,"iterations":1,"gradient_norm":None,"delta_h":[float(proto-h)],"intervention_norm":float(np.linalg.norm(ds)),"read_after":float(read_chart(chart,np.array([[float(proto)]],dtype=np.float64))[0])}
