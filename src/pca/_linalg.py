import numpy as np


def as_float64(values, name="values"):
    values = np.asarray(values)
    if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must be numeric.")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    return values


def pca_from_centered(values, method="svd", rel_tol=None, ddof=0):
    values = as_float64(values, "centered data")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Expected at least two samples and one feature.")
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1.")
    if method not in ("svd", "gram", "covariance"):
        raise ValueError("Unknown PCA decomposition method.")

    sample_count, feature_count = values.shape
    denominator = sample_count - ddof
    if rel_tol is None:
        rel_tol = max(sample_count, feature_count) * np.finfo(float).eps
    if not np.isfinite(rel_tol) or rel_tol < 0:
        raise ValueError("rel_tol must be non-negative.")

    if method == "svd":
        _, singular_values, right_vectors = np.linalg.svd(values, full_matrices=False)
        eigenvalues = singular_values**2 / denominator
        components = right_vectors
    elif method == "gram":
        gram = values @ values.T
        gram = (gram + gram.T) / 2
        eigenvalues, left_vectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        left_vectors = left_vectors[:, order]
        positive = eigenvalues > 0
        eigenvalues = eigenvalues[positive]
        left_vectors = left_vectors[:, positive]
        components = (left_vectors.T @ values) / np.sqrt(eigenvalues[:, None])
        eigenvalues = eigenvalues / denominator
    else:
        covariance = values.T @ values / denominator
        covariance = (covariance + covariance.T) / 2
        eigenvalues, right_vectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        components = right_vectors[:, order].T

    if len(eigenvalues) == 0 or eigenvalues[0] <= 0:
        raise ValueError("The centered data contain no measurable variance.")
    keep = eigenvalues > rel_tol * eigenvalues[0]
    eigenvalues = eigenvalues[keep]
    components = components[keep]
    if len(eigenvalues) == 0:
        raise ValueError("No principal components remain.")
    norms = np.linalg.norm(components, axis=1)
    valid = norms > np.finfo(float).tiny
    return components[valid] / norms[valid, None], eigenvalues[valid]


def choose_k(eigenvalues, n_components):
    eigenvalues = as_float64(eigenvalues, "eigenvalues").reshape(-1)
    if len(eigenvalues) == 0 or np.any(eigenvalues < 0) or eigenvalues.sum() <= 0:
        raise ValueError("Eigenvalues must have a positive sum.")
    if isinstance(n_components, (bool, np.bool_)):
        raise TypeError("n_components cannot be boolean.")
    if isinstance(n_components, (int, np.integer)):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        return min(int(n_components), len(eigenvalues))
    if isinstance(n_components, (float, np.floating)):
        if not 0 < n_components <= 1:
            raise ValueError("Variance fraction must be in (0, 1].")
        cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
        return min(int(np.searchsorted(cumulative, n_components) + 1), len(eigenvalues))
    raise TypeError("Use an integer, a variance fraction, 'all', or None.")
