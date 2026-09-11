from __future__ import annotations

import numpy as np
from experiments.v837_primitive_invention.common.task_interface import Episode, common_prelude, nuisance_vector
from experiments.v837_primitive_invention.tasks import task_by_name
from .utils import HERE, read_json, sha256_json, write_json

ROLE_SEEDS={"FIT_WORDS":list(range(11064,11128)),"SELECT_WORDS":list(range(11128,11192)),"META_WORDS":list(range(11384,11448)),"FINAL_UNSEEN_WORDS":list(range(11448,11512))}


def _episode(obs,target,family,**meta): return Episode(np.asarray(obs,dtype=np.float32),float(target),{"family":family,**meta})
def _iterative_target(x):
    z=0.0
    for v in x: z=.65*z+.35*float(v)
    return float(z)
def _routing_target(c,a,b): return float(a if c>0 else b)
def _clip_add(x,d,lo=-1.0,hi=1.0):
    n=float(np.clip(float(x)+float(d),lo,hi)); return n,n-float(x)


def iterative_word_cases(seed:int,role:str) -> list[dict]:
    base=task_by_name("iterative_state").generate(int(seed),"development"); x=np.asarray(base.observations[1:,0],dtype=float); n=len(x)
    e=0;m=max(0,(n-1)//2);l=n-1; out=[]
    def add(name,mods):
        xp=x.copy(); actual=[]
        for j,d in mods:
            nv,ad=_clip_add(xp[j],d);xp[j]=nv;actual.append((int(j),float(ad)))
        obs=base.observations.copy();obs[1:,0]=xp
        out.append({"name":name,"seed":seed,"role":role,"x":xp.tolist(),"actual_mods":actual,"episode":_episode(obs,_iterative_target(xp),"iterative_state",length=n),"oracle":_iterative_target(xp)})
    sign=lambda j,mag: -float(mag) if x[j]>0 else float(mag)
    if role=="FIT_WORDS":
        add("single_early",[(e,sign(e,1.0))]);add("single_middle",[(m,sign(m,.5))]);add("single_late",[(l,sign(l,1.0))]);
        j=min(max(0,m),max(0,n-2));add("two_adjacent",[(j,sign(j,.5)),(j+1,sign(j+1,.5))])
    elif role=="SELECT_WORDS":
        add("two_separated",[(e,sign(e,1.0)),(l,sign(l,.5))]);add("two_opposite_sign",[(e,sign(e,1.0)),(m,-sign(m,.5))])
    elif role=="META_WORDS": add("three_seen_magnitudes",[(e,sign(e,1.0)),(m,sign(m,.5)),(l,sign(l,1.0))])
    elif role=="FINAL_UNSEEN_WORDS":
        add("three_novel_order",[(l,sign(l,.5)),(e,sign(e,1.0)),(m,sign(m,.5))]);add("four_step",[(min(i,n-1),sign(min(i,n-1),.5 if i%2 else 1.0)) for i in range(min(4,n))]);
        add("four_alternating_sign",[(min(i,n-1),(sign(min(i,n-1),.5) if i%2==0 else -sign(min(i,n-1),.5))) for i in range(min(4,n))]);
        q=max(0,min(n-1,n//3));add("novel_time_locations",[(q,sign(q,1.0))]);add("midpoint_magnitude",[(m,sign(m,.75))])
    else: raise KeyError(role)
    return out


def routing_word_cases(seed:int,role:str) -> list[dict]:
    base=task_by_name("conditional_routing").generate(int(seed),"development"); c=float(base.observations[1,0]);a=float(base.observations[2,1]);b=float(base.observations[3,2]);out=[]
    def add(name,cc,aa,bb):
        obs=base.observations.copy();obs[1,0]=cc;obs[2,1]=aa;obs[3,2]=bb;t=_routing_target(cc,aa,bb)
        out.append({"name":name,"seed":seed,"role":role,"control":float(cc),"A":float(aa),"B":float(bb),"episode":_episode(obs,t,"conditional_routing",control=float(cc)),"oracle":t})
    da=.4 if a<=.45 else -.4; db=-.4 if b>=-.45 else .4
    aa=float(np.clip(a+da,-.85,.85));bb=float(np.clip(b+db,-.85,.85))
    if role=="FIT_WORDS":
        add("control_only",-c,a,b);add("payload_A_only",c,aa,b);add("payload_B_only",c,a,bb);add("control_x_A",-c,aa,b);add("control_x_B",-c,a,bb)
    elif role=="SELECT_WORDS":
        add("A_x_B",c,aa,bb);add("opposite_sign_pairs",c,float(np.clip(a-da,-.85,.85)),float(np.clip(b-db,-.85,.85)))
    elif role=="META_WORDS": add("control_plus_payload_held_sign",-c,aa,b)
    elif role=="FINAL_UNSEEN_WORDS":
        add("control_A_B_triple",-c,aa,bb);add("unseen_known_magnitude_combo",c,float(np.clip(a-da,-.85,.85)),bb)
        amid=float(np.clip(a+(.2 if a<0 else -.2),-.85,.85));bmid=float(np.clip(b+(.2 if b<0 else -.2),-.85,.85));add("midpoint_payload_magnitude",c,amid,bmid)
        # Ordering is not a free semantic degree of this benchmark; preserve semantic channels and record the word as a no-op order permutation.
        add("swapped_order",c,a,b)
    else: raise KeyError(role)
    return out


def recall_max_stream(seed:int):
    rng=np.random.default_rng(int(seed)); _=int(rng.integers(4,13)); value=float(rng.choice([-1.0,1.0])); rows=[common_prelude(rng)]
    present=nuisance_vector(rng,.30);present[0]=value;rows.append(present)
    distractors=[]
    for _i in range(12):
        d=nuisance_vector(rng,.45);d[0]=float(rng.choice([-1.0,1.0]))*.35;distractors.append(d)
    query=nuisance_vector(rng,.30);query[5]=float(rng.normal(0.0,.30))
    return value,rows,distractors,query

def recall_word_cases(seed:int,role:str) -> list[dict]:
    delays={"FIT_WORDS":[4,6,8],"SELECT_WORDS":[5,7],"META_WORDS":[9],"FINAL_UNSEEN_WORDS":[10,11,12]}[role];value,rows,ds,q=recall_max_stream(seed);out=[]
    for n in delays:
        obs=np.stack(rows+ds[:n]+[q]);out.append({"name":f"delay_{n}","seed":seed,"role":role,"delay":n,"value":value,"episode":_episode(obs,value,"delayed_recall",delay=n),"oracle":value})
    return out


def freeze_word_manifest() -> dict:
    p=read_json(HERE/"raw/frozen_operator_word_partitions.json")
    payload={"version":"V837ar","partition_sha256":p["partition_sha256"],"role_seeds":ROLE_SEEDS,"final_opened_before_freeze":False}
    payload["manifest_sha256"]=sha256_json(payload);write_json(HERE/"diagnostics/word_partition_integrity.json",payload);return payload
