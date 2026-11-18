"""Question-only hidden states (extract_hidden.py output) as feature vectors.

``extract_hidden.py`` writes, per split, an ``.npz`` with ``qids``, ``last`` and
``mean`` arrays of shape (N, L, H) and the kept ``layers``. This module turns
one (pooling, layer) slice into a short vector per question:

  * the projection (PCA) is fitted on TRAINING questions only and then applied
    unchanged to calibration and test questions, so nothing about calibration
    or test enters the features;
  * plain numpy SVD, serialisable to JSON like every other fitted object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["load_hidden", "PCA", "hidden_extras"]


def load_hidden(path, pooling: str = "mean", layer: int = -1) -> dict[str, np.ndarray]:
    """{qid: vector} for one pooling ("mean" or "last") at one kept layer.
    ``layer`` indexes the KEPT layers (-1 = the deepest one kept)."""
    if pooling not in ("mean", "last"):
        raise ValueError(f"pooling must be 'mean' or 'last', not {pooling!r}")
    with np.load(path, allow_pickle=False) as z:
        qids = [str(q) for q in z["qids"]]
        arr = np.asarray(z[pooling][:, layer, :], dtype=np.float64)
    if len(set(qids)) != len(qids):
        raise ValueError(f"{path}: duplicate question ids")
    return dict(zip(qids, arr))


@dataclass
class PCA:
    dims: int = 32
    mean: np.ndarray | None = None
    comps: np.ndarray | None = None      # (dims, H)
    scale: np.ndarray | None = None      # per-component std on train, so features are O(1)

    def fit(self, X) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) < 2:
            raise ValueError("PCA needs a 2-D array with at least two rows")
        self.mean = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - self.mean, full_matrices=False)
        k = min(self.dims, vt.shape[0])
        self.comps = vt[:k]
        proj = (X - self.mean) @ self.comps.T
        sd = proj.std(axis=0)
        self.scale = np.where(sd < 1e-12, 1.0, sd)
        return self

    def transform(self, X) -> np.ndarray:
        if self.comps is None:
            raise RuntimeError("PCA is not fitted")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return ((X - self.mean) @ self.comps.T) / self.scale

    def to_dict(self) -> dict:
        return {"dims": self.dims, "mean": self.mean.tolist(), "comps": self.comps.tolist(),
                "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "PCA":
        p = cls(dims=d["dims"])
        p.mean, p.comps, p.scale = (np.asarray(d[k], dtype=np.float64) for k in ("mean", "comps", "scale"))
        return p


def hidden_extras(train_paths, other_paths, train_qids, dims: int = 32, pooling: str = "mean",
                  layer: int = -1) -> tuple[dict[str, list[float]], PCA]:
    """Projected hidden-state vectors for every question in the given files.

    The projection is fitted on ``train_qids`` (which must all be present in
    ``train_paths``) and applied to everything. Returns ({qid: vector}, pca).
    """
    train, other = {}, {}
    for p in train_paths:
        train.update(load_hidden(p, pooling, layer))
    for p in other_paths:
        other.update(load_hidden(p, pooling, layer))
    missing = [q for q in train_qids if q not in train]
    if missing:
        raise ValueError(f"{len(missing)} training questions have no hidden state (first: {missing[:3]})")
    pca = PCA(dims).fit(np.stack([train[q] for q in train_qids]))
    allv = {**other, **train}
    qids = list(allv)
    proj = pca.transform(np.stack([allv[q] for q in qids]))
    return {q: proj[i].tolist() for i, q in enumerate(qids)}, pca
