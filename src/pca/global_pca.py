import numpy as np

from src.pca._linalg import as_float64, choose_k, pca_from_centered


class GlobalPCA:
    def __init__(self, n_components=0.95, method="svd", residual="squared", ddof=0, rel_tol=None):
        if method not in ("svd", "gram", "covariance"):
            raise ValueError("Unknown PCA decomposition method.")
        if residual not in ("squared", "abs"):
            raise ValueError("residual must be squared or abs.")
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1.")
        self.n_components = n_components
        self.method = method
        self.residual = residual
        self.ddof = ddof
        self.rel_tol = rel_tol
        self._fitted = False

    def fit(self, images):
        images = as_float64(images, "images")
        if images.ndim not in (3, 4) or images.shape[0] < 2:
            raise ValueError("Expected at least two images shaped (N,H,W) or (N,C,H,W).")
        if any(size < 1 for size in images.shape[1:]):
            raise ValueError("Image dimensions must be positive.")
        self.image_shape_ = tuple(images.shape[1:])
        flat_images = images.reshape(images.shape[0], -1)
        self.n_train_ = len(flat_images)
        self.mean_ = flat_images.mean(axis=0)
        centered = flat_images - self.mean_
        components, eigenvalues = pca_from_centered(centered, self.method, self.rel_tol, self.ddof)
        self.rank_ = len(eigenvalues)
        self.eigenvalues_all_ = eigenvalues.copy()
        self.total_variance_ = float(eigenvalues.sum())
        count = len(eigenvalues) if self.n_components in (None, "all") else choose_k(eigenvalues, self.n_components)
        self._set_components(components, eigenvalues, count)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit before using the model.")

    def _set_components(self, components, eigenvalues, count):
        if not 1 <= int(count) <= len(eigenvalues):
            raise ValueError("Invalid number of principal components.")
        self.components_ = components[:count].copy()
        self.explained_variance_ = eigenvalues[:count].copy()
        self.explained_variance_ratio_ = self.explained_variance_ / self.total_variance_
        self.n_components_ = int(count)

    def _flatten(self, images):
        self._check_fitted()
        images = as_float64(images, "images")
        if images.ndim != len(self.image_shape_) + 1 or tuple(images.shape[1:]) != self.image_shape_:
            raise ValueError("Input image dimensions do not match training data.")
        return images.reshape(images.shape[0], -1)

    def transform(self, images):
        return (self._flatten(images) - self.mean_) @ self.components_.T

    def inverse_transform(self, codes):
        self._check_fitted()
        codes = as_float64(codes, "codes")
        if codes.ndim != 2 or codes.shape[1] != self.n_components_:
            raise ValueError("Invalid PCA code shape.")
        return (codes @ self.components_ + self.mean_).reshape(-1, *self.image_shape_)

    def reconstruct(self, images):
        return self.inverse_transform(self.transform(images))

    def truncate(self, count):
        self._check_fitted()
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise TypeError("count must be an integer.")
        if not 1 <= count <= self.n_components_:
            raise ValueError("count exceeds fitted component count.")
        model = GlobalPCA(count, self.method, self.residual, self.ddof, self.rel_tol)
        model.image_shape_ = self.image_shape_
        model.n_train_ = self.n_train_
        model.mean_ = self.mean_
        model.rank_ = self.rank_
        model.eigenvalues_all_ = self.eigenvalues_all_
        model.total_variance_ = self.total_variance_
        model._set_components(self.components_, self.explained_variance_, count)
        model._fitted = True
        return model

    def anomaly_map(self, images):
        images = self._flatten(images).reshape(-1, *self.image_shape_)
        residual = images - self.reconstruct(images)
        error = residual**2 if self.residual == "squared" else np.abs(residual)
        if len(self.image_shape_) == 3:
            error = error.mean(axis=1)
        return error.astype(np.float32)

    @staticmethod
    def image_score(anomaly_maps, reduce="max"):
        maps = as_float64(anomaly_maps, "anomaly maps")
        if maps.ndim < 2:
            raise ValueError("Anomaly maps must include batch and spatial dimensions.")
        flat = maps.reshape(maps.shape[0], -1)
        if reduce == "max":
            return flat.max(axis=1)
        if reduce == "mean":
            return flat.mean(axis=1)
        if reduce == "top_1pct":
            count = max(1, int(np.ceil(flat.shape[1] * 0.01)))
            return np.partition(flat, -count, axis=1)[:, -count:].mean(axis=1)
        raise ValueError("reduce must be max, mean or top_1pct.")
