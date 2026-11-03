#!/usr/bin/env python3
"""Compute geometry-based embedding quality baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def effective_rank(x: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(x, compute_uv=False)
    eigenvalues = singular_values**2
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    probabilities = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.exp(entropy))


def uniformity(x: np.ndarray, temperature: float = 2.0, n_samples: int = 5000) -> float:
    if len(x) > n_samples:
        rng = np.random.default_rng(42)
        x = x[rng.choice(len(x), n_samples, replace=False)]
    squared_norms = np.sum(x**2, axis=1, keepdims=True)
    squared_distances = squared_norms + squared_norms.T - 2 * x @ x.T
    upper = squared_distances[np.triu_indices(len(x), k=1)]
    return float(np.log(np.mean(np.exp(-temperature * upper))))


def isoscore(x: np.ndarray) -> float:
    try:
        from IsoScore import IsoScore
    except ImportError:
        return float("nan")
    return float(IsoScore.IsoScore(x))


def silhouette(x: np.ndarray, n_samples: int = 2000) -> float:
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        return float("nan")

    if len(x) > n_samples:
        rng = np.random.default_rng(42)
        x = x[rng.choice(len(x), n_samples, replace=False)]

    best = float("nan")
    for k in (5, 10, 20):
        if k >= len(x):
            continue
        labels = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42).fit_predict(x)
        score = float(silhouette_score(x, labels))
        if np.isnan(best) or score > best:
            best = score
    return best


def twonn_dimension(x: np.ndarray, n_samples: int = 2000) -> float:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        return float("nan")

    if len(x) > n_samples:
        rng = np.random.default_rng(42)
        x = x[rng.choice(len(x), n_samples, replace=False)]
    distances, _ = NearestNeighbors(n_neighbors=3).fit(x).kneighbors(x)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = (r1 > 1e-10) & (r2 > 1e-10)
    mu = r2[valid] / r1[valid]
    return float(len(mu) / np.sum(np.log(mu)))


def load_embedding(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else ("embeddings" if "embeddings" in data else list(data.keys())[0])
        return np.asarray(data[key])
    return np.asarray(np.load(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute geometry baselines for embedding directories.")
    parser.add_argument("--embedding-dir", "--embedding_dir", type=Path, required=True)
    parser.add_argument("--embedding-file", "--embedding_file", default="generation_embeddings.npy")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", help="Optional model subdirectories to evaluate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    models = args.models or sorted(path.name for path in args.embedding_dir.iterdir() if path.is_dir())
    results = []

    for model in models:
        path = args.embedding_dir / model / args.embedding_file
        if not path.exists():
            print(f"Skipping missing embedding: {path}")
            continue
        x = load_embedding(path).astype(np.float32)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        record = {
            "model": model,
            "n_samples": int(x.shape[0]),
            "dim": int(x.shape[1]),
            "effective_rank": effective_rank(x),
            "uniformity": uniformity(x),
            "isoscore": isoscore(x),
            "silhouette": silhouette(x),
            "twonn_dim": twonn_dimension(x),
        }
        results.append(record)
        print(json.dumps(record, indent=2))

    output = {
        "embedding_dir": str(args.embedding_dir),
        "embedding_file": args.embedding_file,
        "results": results,
    }
    output_path = args.output or (args.embedding_dir / "embedding_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
