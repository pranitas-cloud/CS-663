import numpy as np
import pytest

from src.pca.global_pca import GlobalPCA


def make_data(seed=4, sample_count=32, image_shape=(7, 7), rank=5):
    rng = np.random.default_rng(seed)
    feature_count = np.prod(image_shape)
    basis = np.linalg.qr(rng.normal(size=(feature_count, rank)))[0]
    coefficients = rng.normal(size=(sample_count, rank)) * np.arange(rank, 0, -1)
    images = coefficients @ basis.T + 0.4
    images += 1e-3 * rng.normal(size=images.shape)
    return images.reshape(sample_count, *image_shape)


@pytest.fixture
def images():
    return make_data()


def test_mean_covariance_and_rank(images):
    model = GlobalPCA("all").fit(images)
    flat = images.reshape(len(images), -1)
    centered = flat - flat.mean(axis=0)
    covariance = centered.T @ centered / len(images)

    np.testing.assert_allclose(model.mean_, flat.mean(axis=0), atol=1e-12)
    expected = np.linalg.eigvalsh(covariance)[::-1][:model.rank_]
    np.testing.assert_allclose(model.eigenvalues_all_, expected, rtol=1e-7)
    assert model.rank_ <= len(images) - 1
    assert model.total_variance_ == pytest.approx(np.trace(covariance), rel=1e-8)


@pytest.mark.parametrize("method", ["svd", "gram", "covariance"])
def test_pca_methods_agree(images, method):
    reference = GlobalPCA(5, "covariance").fit(images)
    model = GlobalPCA(5, method).fit(images)

    np.testing.assert_allclose(
        reference.explained_variance_, model.explained_variance_, rtol=1e-7
    )
    np.testing.assert_allclose(
        reference.components_.T @ reference.components_,
        model.components_.T @ model.components_,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        reference.reconstruct(images), model.reconstruct(images), atol=1e-7
    )


@pytest.mark.parametrize("method", ["svd", "gram", "covariance"])
def test_eigenvectors_and_orthonormality(images, method):
    model = GlobalPCA(5, method).fit(images)
    flat = images.reshape(len(images), -1)
    centered = flat - flat.mean(axis=0)
    covariance = centered.T @ centered / len(images)

    for eigenvalue, component in zip(
        model.explained_variance_, model.components_
    ):
        np.testing.assert_allclose(
            covariance @ component, eigenvalue * component, atol=1e-8
        )

    np.testing.assert_allclose(
        model.components_ @ model.components_.T, np.eye(5), atol=1e-8
    )


def test_reconstruction_and_anomaly_maps(images):
    model = GlobalPCA("all").fit(images)
    reconstruction = model.reconstruct(images)
    scores, maps = model.predict(images)

    assert reconstruction.shape == images.shape
    assert maps.shape == images.shape
    assert scores.shape == (len(images),)
    assert np.isfinite(maps).all()
    assert maps.min() >= 0
    assert np.mean((images - reconstruction) ** 2) < 1e-8


def test_mse_vs_k(images):
    model = GlobalPCA(0.99).fit(images)
    actual = model.mse_vs_k(images)
    expected = [
        np.mean((images - model.truncate(k).reconstruct(images)) ** 2)
        for k in range(1, model.n_components_ + 1)
    ]

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)


def test_multichannel_anomaly_map():
    images = np.random.default_rng(1).normal(size=(12, 3, 5, 5))
    model = GlobalPCA(0.8).fit(images)

    assert model.anomaly_map(images[:2]).shape == (2, 5, 5)


def test_save_and_load(tmp_path, images):
    model = GlobalPCA(0.9, "gram", "abs", 0).fit(images)
    path = tmp_path / "model.npz"
    model.save(path, category="toy")
    loaded = GlobalPCA.load(path)

    np.testing.assert_allclose(model.reconstruct(images), loaded.reconstruct(images))
    assert loaded.method == "gram"
    assert loaded.residual == "abs"
    assert loaded.ddof == 0


def test_invalid_inputs(images):
    with pytest.raises(ValueError):
        GlobalPCA(method="unknown")
    with pytest.raises(ValueError):
        GlobalPCA(ddof=2)
    with pytest.raises(ValueError):
        GlobalPCA().fit(np.ones((1, 4, 4)))
    with pytest.raises(ValueError):
        GlobalPCA().fit(np.ones((5, 4, 4)))
    with pytest.raises(ValueError):
        GlobalPCA().fit(np.full((5, 4, 4), np.nan))
