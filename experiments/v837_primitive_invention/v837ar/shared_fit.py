from __future__ import annotations
import numpy as np
RIDGE=1e-6

def equal_organism_weights(organism_ids):
    ids=np.asarray(organism_ids);uniq=np.unique(ids);w=np.zeros(len(ids),float)
    for u in uniq:
        m=ids==u;w[m]=1.0/(len(uniq)*int(m.sum()))
    return w

def ridge_fit(X,y,organism_ids,lam=RIDGE):
    X=np.asarray(X,float);y=np.asarray(y,float);w=equal_organism_weights(organism_ids);sw=np.sqrt(w)
    A=X*sw[:,None];b=y*sw
    return np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]),A.T@b)
def per_organism_fit(X,y,organism_ids,lam=RIDGE):
    ids=np.asarray(organism_ids);return {str(u):np.linalg.solve(np.asarray(X)[ids==u].T@np.asarray(X)[ids==u]+lam*np.eye(np.asarray(X).shape[1]),np.asarray(X)[ids==u].T@np.asarray(y)[ids==u]).tolist() for u in np.unique(ids)}
