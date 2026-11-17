"""NumPy-only PCA linear algebra. ddof=0 is the lecture's 1/N covariance."""
import numpy as np

def as_float64(x, name="x"):
    x=np.asarray(x)
    if x.dtype==np.bool_ or not np.issubdtype(x.dtype,np.number): raise TypeError(f"{name} must be numeric.")
    x=x.astype(np.float64,copy=False)
    if not np.isfinite(x).all(): raise ValueError(f"{name} contains NaN or inf.")
    return x

def pca_from_centered(x, method="svd", rel_tol=None, ddof=0):
    x=as_float64(x,"centered")
    if x.ndim!=2: raise ValueError("centered must have shape (N,D).")
    n,d=x.shape
    if n<2 or d<1: raise ValueError("Need at least two samples and one feature.")
    if ddof not in (0,1): raise ValueError("ddof must be 0 or 1.")
    if method not in ("svd","gram","covariance"): raise ValueError("Unknown PCA method.")
    den=n-ddof
    if rel_tol is None: rel_tol=max(n,d)*np.finfo(float).eps
    if not np.isfinite(rel_tol) or rel_tol<0: raise ValueError("rel_tol must be non-negative.")
    if method=="svd":
        _,s,vt=np.linalg.svd(x,full_matrices=False); vals=s*s/den; vecs=vt
    elif method=="gram":
        g=x@x.T; g=(g+g.T)*0.5
        w,u=np.linalg.eigh(g); order=np.argsort(w)[::-1]; w=w[order]; u=u[:,order]
        positive=w>0; w=w[positive]; u=u[:,positive]
        vals=w/den; vecs=(u.T@x)/np.sqrt(w[:,None])
    else:
        c=(x.T@x)/den; c=(c+c.T)*0.5
        w,v=np.linalg.eigh(c); order=np.argsort(w)[::-1]; vals=w[order]; vecs=v[:,order].T
    if not len(vals) or vals[0]<=0: raise ValueError("The centred data contain no measurable variance.")
    keep=vals>rel_tol*vals[0]; vals=vals[keep]; vecs=vecs[keep]
    if not len(vals): raise ValueError("No components remain after rank cutoff.")
    norms=np.linalg.norm(vecs,axis=1); valid=norms>np.finfo(float).tiny
    return vecs[valid]/norms[valid,None], vals[valid]

def choose_k(eigenvalues,n_components):
    v=as_float64(eigenvalues,"eigenvalues").reshape(-1)
    if not len(v) or np.any(v<0) or v.sum()<=0: raise ValueError("Eigenvalues must have positive total variance.")
    if isinstance(n_components,(bool,np.bool_)): raise TypeError("n_components cannot be bool.")
    if isinstance(n_components,(int,np.integer)):
        if n_components<1: raise ValueError("Integer n_components must be >=1.")
        return min(int(n_components),len(v))
    if isinstance(n_components,(float,np.floating)):
        if not 0<float(n_components)<=1: raise ValueError("Variance fraction must lie in (0,1].")
        return min(int(np.searchsorted(np.cumsum(v)/v.sum(),float(n_components))+1),len(v))
    raise TypeError("Use an integer, variance fraction, 'all', or None.")
