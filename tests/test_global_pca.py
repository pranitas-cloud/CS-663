import numpy as np
import pytest
from src.pca.global_pca import GlobalPCA

def data(seed=4,n=32,shape=(7,7),rank=5):
    r=np.random.default_rng(seed); d=np.prod(shape); q=np.linalg.qr(r.normal(size=(d,rank)))[0]
    z=r.normal(size=(n,rank))*np.arange(rank,0,-1)
    return (z@q.T+.4+1e-3*r.normal(size=(n,d))).reshape(n,*shape)

@pytest.fixture
def x(): return data()

def test_mean_covariance_and_rank(x):
    m=GlobalPCA("all").fit(x); a=x.reshape(len(x),-1); c=(a-a.mean(0)).T@(a-a.mean(0))/len(x)
    np.testing.assert_allclose(m.mean_,a.mean(0),atol=1e-12)
    np.testing.assert_allclose(m.eigenvalues_all_,np.linalg.eigvalsh(c)[::-1][:m.rank_],rtol=1e-7)
    assert m.rank_<=len(x)-1
    assert m.total_variance_==pytest.approx(np.trace(c),rel=1e-8)

@pytest.mark.parametrize("method",["svd","gram","covariance"])
def test_three_routes_match(x,method):
    a=GlobalPCA(5,"covariance").fit(x); b=GlobalPCA(5,method).fit(x)
    np.testing.assert_allclose(a.explained_variance_,b.explained_variance_,rtol=1e-7)
    np.testing.assert_allclose(a.components_.T@a.components_,b.components_.T@b.components_,atol=1e-7)
    np.testing.assert_allclose(a.reconstruct(x),b.reconstruct(x),atol=1e-7)

@pytest.mark.parametrize("method",["svd","gram","covariance"])
def test_eigen_equation_and_orthonormality(x,method):
    m=GlobalPCA(5,method).fit(x); a=x.reshape(len(x),-1); xc=a-a.mean(0); c=xc.T@xc/len(x)
    for lam,v in zip(m.explained_variance_,m.components_): np.testing.assert_allclose(c@v,lam*v,atol=1e-8)
    np.testing.assert_allclose(m.components_@m.components_.T,np.eye(5),atol=1e-8)

def test_maps_scores_and_reconstruction(x):
    m=GlobalPCA("all").fit(x); rec=m.reconstruct(x); scores,maps=m.predict(x)
    assert rec.shape==x.shape and maps.shape==x.shape and scores.shape==(len(x),)
    assert np.isfinite(maps).all() and maps.min()>=0
    assert np.mean((x-rec)**2)<1e-8

def test_mse_vs_k_and_truncate(x):
    m=GlobalPCA(.99).fit(x); got=m.mse_vs_k(x)
    expected=[np.mean((x-m.truncate(k).reconstruct(x))**2) for k in range(1,m.n_components_+1)]
    np.testing.assert_allclose(got,expected,rtol=1e-7,atol=1e-10)

def test_multichannel_maps():
    x=np.random.default_rng(1).normal(size=(12,3,5,5)); m=GlobalPCA(.8).fit(x)
    assert m.anomaly_map(x[:2]).shape==(2,5,5)

def test_save_load(tmp_path,x):
    m=GlobalPCA(.9,"gram","abs",0).fit(x); p=tmp_path/"m.npz"; m.save(p,category="toy"); q=GlobalPCA.load(p)
    np.testing.assert_allclose(m.reconstruct(x),q.reconstruct(x))
    assert q.method=="gram" and q.residual=="abs" and q.ddof==0

def test_errors(x):
    with pytest.raises(ValueError): GlobalPCA(method="bad")
    with pytest.raises(ValueError): GlobalPCA(ddof=2)
    with pytest.raises(ValueError): GlobalPCA().fit(np.ones((1,4,4)))
    with pytest.raises(ValueError): GlobalPCA().fit(np.ones((5,4,4)))
    with pytest.raises(ValueError): GlobalPCA().fit(np.full((5,4,4),np.nan))
    with pytest.raises(ValueError): GlobalPCA().fit(np.ones((5,4,4)))
