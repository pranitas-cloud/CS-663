import numpy as np

from src.pca._linalg import as_float64, choose_k, pca_from_centered


class GlobalPCA:
    def __init__(
        self,
        n_components=0.95,
        method="svd",
        residual="squared",
        ddof=0,
        rel_tol=None,
    ):
        if method not in ("svd", "gram", "covariance"):
            raise ValueError("method must be svd, gram, or covariance.")
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
            raise ValueError("Expected at least two images shaped (N, H, W) or (N, C, H, W).")
        if any(size < 1 for size in images.shape[1:]):
            raise ValueError("Image dimensions must be positive.")

        self.image_shape_ = tuple(images.shape[1:])
        flat_images = images.reshape(images.shape[0], -1)
        self.n_train_ = len(flat_images)
        self.mean_ = flat_images.mean(axis=0)

        centered = flat_images - self.mean_
        components, eigenvalues = pca_from_centered(
            centered, self.method, self.rel_tol, self.ddof
        )

        self.rank_ = len(eigenvalues)
        self.eigenvalues_all_ = eigenvalues.copy()
        self.total_variance_ = float(eigenvalues.sum())

        if self.n_components is None or self.n_components == "all":
            component_count = len(eigenvalues)
        else:
            component_count = choose_k(eigenvalues, self.n_components)

        self._set_components(components, eigenvalues, component_count)
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
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.total_variance_
        )
        self.n_components_ = int(count)

    def _flatten(self, images):
        self._check_fitted()
        images = as_float64(images, "images")

        if images.ndim != len(self.image_shape_) + 1:
            raise ValueError("Input image dimensions do not match the training data.")
        if tuple(images.shape[1:]) != self.image_shape_:
            raise ValueError("Input image dimensions do not match the training data.")

        return images.reshape(images.shape[0], -1)

    def transform(self, images):
        centered = self._flatten(images) - self.mean_
        return centered @ self.components_.T

    def inverse_transform(self, codes):
        self._check_fitted()
        codes = as_float64(codes, "codes")

        if codes.ndim != 2 or codes.shape[1] != self.n_components_:
            raise ValueError("Invalid PCA code shape.")

        reconstructed = codes @ self.components_ + self.mean_
        return reconstructed.reshape(-1, *self.image_shape_)

    def reconstruct(self, images):
        return self.inverse_transform(self.transform(images))

    def truncate(self, count):
        self._check_fitted()

        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise TypeError("count must be an integer.")
        if not 1 <= count <= self.n_components_:
            raise ValueError("count exceeds the fitted component count.")

        model = GlobalPCA(
            n_components=count,
            method=self.method,
            residual=self.residual,
            ddof=self.ddof,
            rel_tol=self.rel_tol,
        )
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

        if self.residual == "squared":
            error = residual**2
        else:
            error = np.abs(residual)

        if len(self.image_shape_) == 3:
            error = error.mean(axis=1)

        return error.astype(np.float32)

    @staticmethod
    def image_score(anomaly_maps, reduce="max"):
        anomaly_maps = as_float64(anomaly_maps, "anomaly maps")

        if anomaly_maps.ndim < 2:
            raise ValueError("Anomaly maps must include batch and spatial dimensions.")

        flattened = anomaly_maps.reshape(anomaly_maps.shape[0], -1)

        if reduce == "max":
            return flattened.max(axis=1)
        if reduce == "mean":
            return flattened.mean(axis=1)

        raise ValueError("reduce must be max or mean.")

    def predict(self, images, reduce="max"):
        maps = self.anomaly_map(images)
        return self.image_score(maps, reduce), maps

    def mse_vs_k(self, images):
        centered = self._flatten(images) - self.mean_
        codes = centered @ self.components_.T
        total_energy = np.einsum("ij,ij->i", centered, centered)[:, None]
        residual_energy = np.maximum(total_energy - np.cumsum(codes**2, axis=1), 0)
        return (residual_energy / centered.shape[1]).mean(axis=0)

    def mahalanobis_score(self, images):
        codes = self.transform(images)
        return np.sum(codes**2 / self.explained_variance_, axis=1)

    def save(self, path, **metadata):
        self._check_fitted()

        data = {
            "mean": self.mean_,
            "components": self.components_,
            "explained_variance": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_,
            "eigenvalues_all": self.eigenvalues_all_,
            "total_variance": np.asarray(self.total_variance_),
            "image_shape": np.asarray(self.image_shape_),
            "n_train": np.asarray(self.n_train_),
            "residual": np.asarray(self.residual),
            "ddof": np.asarray(self.ddof),
            "method": np.asarray(self.method),
            "rel_tol": np.asarray(-1.0 if self.rel_tol is None else self.rel_tol),
        }

        for key, value in metadata.items():
            if key in data or key.startswith("meta_"):
                raise ValueError(f"Reserved metadata key: {key}")
            data[f"meta_{key}"] = np.asarray(value)

        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            tolerance = float(data["rel_tol"]) if "rel_tol" in data else -1.0
            model = cls(
                n_components=int(data["components"].shape[0]),
                method=str(data["method"]) if "method" in data else "svd",
                residual=str(data["residual"]),
                ddof=int(data["ddof"]) if "ddof" in data else 0,
                rel_tol=None if tolerance < 0 else tolerance,
            )

            model.mean_ = data["mean"].copy()
            model.components_ = data["components"].copy()
            model.explained_variance_ = data["explained_variance"].copy()
            model.explained_variance_ratio_ = data["explained_variance_ratio"].copy()
            model.eigenvalues_all_ = (
                data["eigenvalues_all"].copy()
                if "eigenvalues_all" in data
                else model.explained_variance_.copy()
            )
            model.total_variance_ = float(data["total_variance"])
            model.image_shape_ = tuple(map(int, data["image_shape"]))
            model.n_train_ = int(data["n_train"])
            model.rank_ = len(model.eigenvalues_all_)
            model.n_components_ = len(model.components_)
            model.n_components = model.n_components_
            model._fitted = True

        return model
