from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path
from typing import Sequence

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from experiments.v837_primitive_invention.v837ak.boundary_traces import FullProbeTrace, active_mask, run_full_probe
from experiments.v837_primitive_invention.v837ak.reconstruct_organisms import load_reconstructed_model
from experiments.v837_primitive_invention.v837ak.utils import HERE, covariance_spectrum, occurrence_id, participation_ratio, read_json, safe_spectrum

CACHE=HERE/"raw/cache/fingerprints"
DISCOVERY_SEEDS=list(range(20000,20064))


def _slice_trace(trace:FullProbeTrace,start:int,end:int)->FullProbeTrace:
    payload={}
    for f in fields(FullProbeTrace):
        tensor=getattr(trace,f.name)
        payload[f.name]=tensor[start:end]
    return FullProbeTrace(**payload)


def _prev_states(states:torch.Tensor)->torch.Tensor:
    prev=torch.zeros_like(states); prev[:,1:]=states[:,:-1]; return prev


def _edge_contributions(model,trace:FullProbeTrace)->dict[tuple[int,int,bool],torch.Tensor]:
    prev_out=torch.zeros_like(trace.outputs); prev_out[:,1:]=trace.outputs[:,:-1]
    result={}
    for idx,e in enumerate(model.graph.edges):
        source=prev_out[:,:,int(e.src),:] if bool(e.recurrent) else trace.outputs[:,:,int(e.src),:]
        result[(int(e.src),int(e.dst),bool(e.recurrent))]=float(model.base.edge_weights[idx].detach().cpu().item())*source
    return result


def _internal_external(model,trace:FullProbeTrace,nodes:Sequence[int],edge_contrib:dict)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    ordered=tuple(sorted(nodes)); s=set(ordered); B,T=trace.states.shape[:2]
    internal=torch.zeros(B,T,len(ordered),4,dtype=trace.states.dtype)
    local={n:i for i,n in enumerate(ordered)}
    for (src,dst,rec),value in edge_contrib.items():
        if src in s and dst in s: internal[:,:,local[dst],:]+=value
    mask=active_mask(trace).to(internal.dtype).unsqueeze(-1).unsqueeze(-1); internal*=mask
    total=trace.messages[:,:,list(ordered),:]; external=total-internal
    internal_term=[]; external_term=[]
    for li,orig in enumerate(ordered):
        wm=model.base.cell_wm[orig].detach().cpu()
        internal_term.append(internal[:,:,li,:]@wm.T); external_term.append(external[:,:,li,:]@wm.T)
    return internal.numpy(),external.numpy(),torch.stack(internal_term,dim=2).numpy(),torch.stack(external_term,dim=2).numpy()


def _corr(a:np.ndarray,b:np.ndarray)->float:
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.size<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return 0.0
    return float(np.corrcoef(a,b)[0,1])


def _lag_cca(x:np.ndarray,y:np.ndarray,ridge:float=1e-6)->np.ndarray:
    d=x.shape[1]
    if x.shape[0]<3:return np.zeros(d)
    x=x-x.mean(0); y=y-y.mean(0)
    cxx=x.T@x/max(1,len(x)-1)+ridge*np.eye(d); cyy=y.T@y/max(1,len(y)-1)+ridge*np.eye(d); cxy=x.T@y/max(1,len(x)-1)
    def invsqrt(c):
        vals,vec=np.linalg.eigh(c); vals=np.clip(vals,ridge,None); return (vec*(1.0/np.sqrt(vals)))@vec.T
    m=invsqrt(cxx)@cxy@invsqrt(cyy)
    return np.clip(np.linalg.svd(m,compute_uv=False),0.0,1.0)


def _transition(x:np.ndarray,y:np.ndarray,ridge:float=1e-6)->tuple[np.ndarray,float,float]:
    d=x.shape[1]
    if len(x)<3:return np.zeros(d),0.0,0.0
    xc=x-x.mean(0); yc=y-y.mean(0)
    A=np.linalg.solve(xc.T@xc+ridge*np.eye(d),xc.T@yc)
    sv=np.linalg.svd(A,compute_uv=False)
    eig=np.linalg.eigvals(A)
    radius=float(np.max(np.abs(eig))) if eig.size else 0.0
    slow=float(np.mean(np.abs(eig)>=0.9)) if eig.size else 0.0
    return sv,radius,slow


def _motif_response(prev:np.ndarray,ext:np.ndarray,glob:np.ndarray,inp:np.ndarray,gate:np.ndarray,target:np.ndarray,ridge:float=1e-6):
    """Fit the frozen joint motif boundary-response model.

    All motif cells participate in one 4k-dimensional state/operator rather
    than assuming independent per-cell blocks. The intercept is represented by
    centering, so the returned matrices correspond to K, B_m, B_g, B_x, B_c.
    """
    z=np.concatenate([prev,ext,glob,inp,gate[:,None]],axis=1).astype(np.float64)
    y=target.astype(np.float64)
    zc=z-z.mean(0); yc=y-y.mean(0)
    beta=np.linalg.solve(zc.T@zc+ridge*np.eye(zc.shape[1]),zc.T@yc)
    d=prev.shape[1]
    return beta[0:d],beta[d:2*d],beta[2*d:3*d],beta[3*d:4*d],beta[4*d:4*d+1]


def _energy(x:np.ndarray)->float:
    if x.size==0:return 0.0
    return float(np.mean(np.sum(np.asarray(x,dtype=np.float64)**2,axis=-1)))


def fingerprint_occurrence(model,trace:FullProbeTrace,nodes:Sequence[int],edge_contrib:dict|None=None)->np.ndarray:
    ordered=tuple(sorted(int(n) for n in nodes)); k=len(ordered); d=4*k
    mask=active_mask(trace).numpy(); states=trace.states[:,:,list(ordered),:].reshape(trace.states.shape[0],trace.states.shape[1],d).numpy()
    cand=trace.candidates[:,:,list(ordered),:].reshape(trace.states.shape[0],trace.states.shape[1],d).numpy(); outputs=trace.outputs[:,:,list(ordered),:].reshape(trace.states.shape[0],trace.states.shape[1],d).numpy()
    sflat=states[mask]; cflat=cand[mask]; oflat=outputs[mask]
    ss=covariance_spectrum(sflat); cs=covariance_spectrum(cflat); ospec=covariance_spectrum(oflat)
    pr=participation_ratio(ss)
    trans_mask=(np.arange(trace.states.shape[1]-1)[None,:] < (trace.lengths.numpy()[:,None]-1))
    x=states[:,:-1][trans_mask]; y=states[:,1:][trans_mask]
    cca=_lag_cca(x,y); tsv,radius,slow=_transition(x,y)
    if edge_contrib is None: edge_contrib=_edge_contributions(model,trace)
    _,external,internal_term,external_term=_internal_external(model,trace,ordered,edge_contrib)
    prev_cells=_prev_states(trace.states)[:,:,list(ordered),:].numpy(); glob_cells=trace.global_terms[:,:,list(ordered),:].numpy()+trace.matched_terms[:,:,list(ordered),:].numpy(); inp_cells=trace.input_terms[:,:,list(ordered),:].numpy(); gate=trace.gates.squeeze(-1).numpy()
    prev_joint=prev_cells.reshape(trace.states.shape[0],trace.states.shape[1],d); ext_joint=external.reshape(trace.states.shape[0],trace.states.shape[1],d); glob_joint=glob_cells.reshape(trace.states.shape[0],trace.states.shape[1],d); inp_joint=inp_cells.reshape(trace.states.shape[0],trace.states.shape[1],d)
    beta=_motif_response(prev_joint[mask],ext_joint[mask],glob_joint[mask],inp_joint[mask],gate[mask],states[mask])
    K=np.linalg.svd(beta[0],compute_uv=False); Bm=np.linalg.svd(beta[1],compute_uv=False); Bg=np.linalg.svd(beta[2],compute_uv=False); Bx=np.linalg.svd(beta[3],compute_uv=False); Bc=float(np.linalg.norm(beta[4]))
    local=trace.local_terms[:,:,list(ordered),:].numpy()[mask]; global_arr=glob_cells[mask]; input_arr=inp_cells[mask]; int_arr=internal_term[mask]; ext_arr=external_term[mask]
    energies=np.array([_energy(local),_energy(int_arr),_energy(ext_arr),_energy(global_arr),_energy(input_arr)],dtype=np.float64); total=float(energies.sum())+1e-12
    ratios=list(energies/total)+[_energy(cflat)/total,_energy(sflat)/total]
    state_norm=np.linalg.norm(sflat,axis=1); cand_norm=np.linalg.norm(cflat,axis=1); gate_flat=gate[mask]
    gate_features=[float(np.mean(gate_flat)),float(np.var(gate_flat)),_corr(gate_flat,state_norm),_corr(gate_flat,cand_norm)]
    vec=np.concatenate([ss,cs,ospec,[pr],cca,tsv,[radius,slow],K,Bm,Bg,Bx,[float(np.mean(Bc))],np.asarray(ratios),np.asarray(gate_features)]).astype(np.float64)
    return np.nan_to_num(vec,nan=0.0,posinf=0.0,neginf=0.0)


def _worker(row:dict)->dict:
    oid=row["organism_id"]; trace=run_full_probe(oid,DISCOVERY_SEEDS,cache_key="discovery64"); model,_,_=load_reconstructed_model(row)
    d1=_slice_trace(trace,0,32); d2=_slice_trace(trace,32,64); e1=_edge_contributions(model,d1); e2=_edge_contributions(model,d2)
    arrays={}
    for k in range(1,11):
        combos=list(itertools.combinations(range(10),k)); ids=[]; masks=[]; v1=[]; v2=[]
        for nodes in combos:
            ids.append(occurrence_id(oid,nodes)); masks.append(sum(1<<n for n in nodes)); v1.append(fingerprint_occurrence(model,d1,nodes,e1)); v2.append(fingerprint_occurrence(model,d2,nodes,e2))
        arrays[f"ids_{k}"]=np.asarray(ids,dtype="U64"); arrays[f"masks_{k}"]=np.asarray(masks,dtype=np.int16); arrays[f"d1_{k}"]=np.stack(v1); arrays[f"d2_{k}"]=np.stack(v2)
    path=CACHE/f"{oid}.npz"; path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**arrays)
    return {"organism_id":oid,"cache":path.relative_to(ROOT).as_posix(),"competent":bool(row["competent"]),"engine":row["engine"],"family":row["family"]}


def generate_all_fingerprints()->dict:
    recon=read_json(HERE/"raw/reconstruction_results.json")
    if recon.get("complete") is not True: raise RuntimeError("AK4 blocked: reconstruction incomplete")
    existing={p.stem for p in CACHE.glob("*.npz")} if CACHE.is_dir() else set(); jobs=[r for r in recon["rows"] if r["organism_id"] not in existing]
    results=[]
    if jobs:
        workers=min(8,max(1,os.cpu_count() or 1),len(jobs))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_worker,row):row for row in jobs}
            for f in as_completed(futures):
                r=f.result(); results.append(r); print(f"fingerprinted {r['engine']} {r['family']} {r['organism_id'][:8]}",flush=True)
    index=[]
    for row in recon["rows"]:
        path=CACHE/f"{row['organism_id']}.npz"
        if not path.is_file():raise RuntimeError("V837AK_FINGERPRINT_CACHE_MISSING")
        with np.load(path) as z:
            counts={str(k):int(z[f"d1_{k}"].shape[0]) for k in range(1,11)}; dims={str(k):int(z[f"d1_{k}"].shape[1]) for k in range(1,11)}
        index.append({"organism_id":row["organism_id"],"engine":row["engine"],"family":row["family"],"competent":bool(row["competent"]),"cache":path.relative_to(ROOT).as_posix(),"counts":counts,"dimensions":dims})
    return {"version":"V837ak","organisms":50,"fingerprints_per_probe":50*1023,"fingerprints_total":2*50*1023,"rows":index}


def load_size_matrix(k:int,which:str="mean",competent_only:bool=False):
    recon=read_json(HERE/"raw/reconstruction_results.json"); meta={r["organism_id"]:r for r in recon["rows"]}
    vectors=[]; records=[]
    for oid,row in sorted(meta.items()):
        if competent_only and not row["competent"]:continue
        with np.load(CACHE/f"{oid}.npz") as z:
            a=z[f"d1_{k}"]; b=z[f"d2_{k}"]; ids=z[f"ids_{k}"]; masks=z[f"masks_{k}"]
            mat=a if which=="d1" else b if which=="d2" else (a+b)/2.0
            for i in range(len(ids)):
                vectors.append(mat[i]); records.append({"occurrence_id":str(ids[i]),"organism_id":oid,"engine":row["engine"],"family":row["family"],"competent":bool(row["competent"]),"nodes":[n for n in range(10) if int(masks[i])&(1<<n)],"size":k})
    return records,np.stack(vectors)


def main()->int:
    payload=generate_all_fingerprints(); print(json.dumps({"organisms":payload["organisms"],"fingerprints_total":payload["fingerprints_total"]},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
