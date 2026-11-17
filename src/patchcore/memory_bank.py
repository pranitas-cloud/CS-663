"""Random-subset feature memory bank and nearest-neighbour scoring."""

import torch


class MemoryBank:
    def __init__(self, max_samples=10000):
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        self.max_samples = max_samples
        self.features = None

    def fit(self, feature_batches):
        batches = list(feature_batches)
        if not batches:
            raise ValueError("At least one feature batch is required.")
        features = torch.cat([
            feature.detach().flatten(2).transpose(1, 2).reshape(-1, feature.shape[1]).cpu()
            for feature in batches
        ])
        if len(features) > self.max_samples:
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(features), generator=generator)[:self.max_samples]
            features = features[indices]
        self.features = torch.nn.functional.normalize(features, dim=1)
        return self

    def query(self, features, chunk_size=2048):
        if self.features is None:
            raise RuntimeError("Call fit() before query().")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        queries = torch.nn.functional.normalize(
            features.flatten(2).transpose(1, 2),
            dim=-1,
        )
        bank = self.features.to(queries.device)
        outputs = []
        for sample in queries:
            chunks = []
            for part in sample.split(chunk_size):
                similarities = part @ bank.T
                chunks.append((1.0 - similarities.max(dim=1).values).clamp_min(0))
            outputs.append(torch.cat(chunks))
        return torch.stack(outputs)
