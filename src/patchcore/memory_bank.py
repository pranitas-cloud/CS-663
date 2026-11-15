"""Coreset-style random memory bank and nearest-neighbour scoring."""
import torch


class MemoryBank:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.features = None

    def fit(self, feature_batches):
        feats = torch.cat([f.detach().flatten(2).transpose(1, 2).reshape(-1, f.shape[1]).cpu() for f in feature_batches])
        if len(feats) > self.max_samples:
            # Reproducible random subset baseline; replace with greedy coreset for final study.
            gen = torch.Generator().manual_seed(42)
            idx = torch.randperm(len(feats), generator=gen)[:self.max_samples]
            feats = feats[idx]
        self.features = torch.nn.functional.normalize(feats, dim=1)
        return self

    def query(self, features, chunk_size=2048):
        if self.features is None:
            raise RuntimeError("Call fit() before query().")
        q = torch.nn.functional.normalize(features.flatten(2).transpose(1, 2), dim=-1)
        bank = self.features.to(q.device)
        outputs = []
        for sample in q:
            chunks = []
            for part in sample.split(chunk_size):
                # cosine distance; clamp for numerical stability
                sim = part @ bank.T
                chunks.append((1.0 - sim.max(dim=1).values).clamp_min(0))
            outputs.append(torch.cat(chunks))
        return torch.stack(outputs)
