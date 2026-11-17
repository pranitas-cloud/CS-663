"""Global image PCA anomaly detector (NumPy only).

Lecture core: training mean, centring, covariance eigensystem, projection,
reconstruction, squared reconstruction residual. SVD/Gram are computational
routes to the same principal subspace; covariance is the direct reference route.
"""
import numpy as np
from src.pca._linalg import as_float64, choose_k, pca_from_centered

class GlobalPCA:
    def __init__(self,n_components=0.95,method="svd",residual="squared",ddof=0,rel_tol=None):
        if method not in ("svd","gram","covariance"): raise ValueError("method must be svd, gram, or covariance.")
        if residual not in ("squared","abs"): raise ValueError("residual must be squared or abs.")
        if ddof not in (0,1): raise ValueError("ddof must be 0 or 1.")
        self.n_components=n_components; self.method=method; self.residual=residual; self.ddof=ddof; self.rel_tol=rel_tol
        self._fitted=False

    def fit(self,images):
        x=as_float64(images,"images")
        if x.ndim not in (3,4) or x.shape[0]<2 or any(s<1 for s in x.shape[1:]):
            raise ValueError("Expected at least two images shaped (N,H,W) or (N,C,H,W).")
        self.image_shape_=tuple(x.shape[1:]); flat=x.reshape(x.shape[0],-1); self.n_train_=len(flat)
        self.mean_=flat.mean(axis=0); centered=flat-self.mean_
        components,values=pca_from_centered(centered,self.method,self.rel_tol,self.ddof)
        self.rank_=len(values); self.eigenvalues_all_=values.copy(); self.total_variance_=float(values.sum())
        k=len(values) if self.n_components is None or self.n_components=="all" else choose_k(values,self.n_components)
        self._set_components(components,values,k); self._fitted=True
        return self

    def _check(self):
        if not self._fitted: raise RuntimeError("Call fit before using the model.")

    def _set_components(self,components,values,k):
        if not 1<=int(k)<=len(values): raise ValueError("k must be within available numerical rank.")
        self.components_=components[:k].copy(); self.explained_variance_=values[:k].copy()
        self.explained_variance_ratio_=self.explained_variance_/self.total_variance_; self.n_components_=int(k)

    def _flat(self,images):
        self._check(); x=as_float64(images,"images")
        if x.ndim!=len(self.image_shape_)+1 or tuple(x.shape[1:])!=self.image_shape_:
            raise ValueError(f"Expected (N,{self.image_shape_}), got {x.shape}.")
        return x.reshape(x.shape[0],-1)

    def truncate(self,k):
        self._check()
        if isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)): raise TypeError("k must be integer.")
        if not 1<=k<=self.n_components_: raise ValueError("k exceeds fitted component count.")
        m=GlobalPCA(k,self.method,self.residual,self.ddof,self.rel_tol)
        m.image_shape_=self.image_shape_; m.n_train_=self.n_train_; m.mean_=self.mean_; m.rank_=self.rank_
        m.eigenvalues_all_=self.eigenvalues_all_; m.total_variance_=self.total_variance_
        m._set_components(self.components_,self.explained_variance_,k); m._fitted=True
        return m

    def transform(self,images): return (self._flat(images)-self.mean_)@self.components_.T

    def inverse_transform(self,codes):
        self._check(); z=as_float64(codes,"codes")
        if z.ndim!=2 or z.shape[1]!=self.n_components_: raise ValueError("Invalid PCA code shape.")
        return (z@self.components_+self.mean_).reshape(-1,*self.image_shape_)

    def reconstruct(self,images): return self.inverse_transform(self.transform(images))

    def anomaly_map(self,images):
        x=self._flat(images).reshape(-1,*self.image_shape_); r=x-self.reconstruct(x)
        e=r**2 if self.residual=="squared" else np.abs(r)
        if len(self.image_shape_)==3: e=e.mean(axis=1)
        return e.astype(np.float32)

    @staticmethod
    def image_score(maps,reduce="max"):
        a=as_float64(maps,"maps")
        if a.ndim<2: raise ValueError("maps must include batch and spatial dimensions.")
        a=a.reshape(a.shape[0],-1)
        if reduce=="max": return a.max(axis=1)
        if reduce=="mean": return a.mean(axis=1)
        raise ValueError("reduce must be max or mean.")

    def predict(self,images,reduce="max"):
        maps=self.anomaly_map(images); return self.image_score(maps,reduce),maps

    def mse_vs_k(self,images):
        c=self._flat(images)-self.mean_; z=c@self.components_.T
        residual=np.maximum(np.einsum("ij,ij->i",c,c)[:,None]-np.cumsum(z*z,axis=1),0)
        return (residual/c.shape[1]).mean(axis=0)

    def mahalanobis_score(self,images):
        z=self.transform(images); return np.sum(z*z/self.explained_variance_,axis=1)

    def save(self,path,**meta):
        self._check()
        data=dict(mean=self.mean_,components=self.components_,explained_variance=self.explained_variance_,
          explained_variance_ratio=self.explained_variance_ratio_,eigenvalues_all=self.eigenvalues_all_,
          total_variance=np.asarray(self.total_variance_),image_shape=np.asarray(self.image_shape_),
          n_train=np.asarray(self.n_train_),residual=np.asarray(self.residual),ddof=np.asarray(self.ddof),
          method=np.asarray(self.method),rel_tol=np.asarray(-1. if self.rel_tol is None else self.rel_tol))
        for k,v in meta.items():
            if k in data or k.startswith("meta_"): raise ValueError(f"Reserved metadata key: {k}")
            data["meta_"+k]=np.asarray(v)
        np.savez_compressed(path,**data)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            tol=float(d["rel_tol"]) if "rel_tol" in d else -1.
            m=cls(n_components=int(d["components"].shape[0]),method=str(d["method"]) if "method" in d else "svd",
                  residual=str(d["residual"]),ddof=int(d["ddof"]) if "ddof" in d else 0,
                  rel_tol=None if tol<0 else tol)
            m.mean_=d["mean"].copy(); m.components_=d["components"].copy()
            m.explained_variance_=d["explained_variance"].copy()
            m.explained_variance_ratio_=d["explained_variance_ratio"].copy()
            m.eigenvalues_all_=d["eigenvalues_all"].copy() if "eigenvalues_all" in d else m.explained_variance_.copy()
            m.total_variance_=float(d["total_variance"]); m.image_shape_=tuple(map(int,d["image_shape"]))
            m.n_train_=int(d["n_train"]); m.rank_=len(m.eigenvalues_all_); m.n_components_=len(m.components_)
            m.n_components=m.n_components_; m._fitted=True
        return m
