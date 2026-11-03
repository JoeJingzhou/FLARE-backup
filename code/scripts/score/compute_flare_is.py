#!/usr/bin/env python3
"""Aggregate FLARE normalizing-flow scores into model-level IS scores.

This script belongs to the FLARE method path, not the EMIR baseline path. It
expects per-pair NLL caches produced by trained marginal and conditional flows.

For a source model U and target model V, FLARE estimates:

    I_s(U -> V) = H(V) - H(V | U)

If the cache stores negative log-likelihoods, this becomes:

    mean(marginal_nll) - mean(conditional_nll)

The pair score is divided by the target embedding dimension, then source-model
scores are aggregated across targets, median by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path):
    import numpy as np

    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def infer_key(keys: list[str], preferred: str | None, candidates: tuple[str, ...]) -> str:
    if preferred:
        if preferred not in keys:
            raise KeyError(f"Requested key {preferred!r} not found. Available keys: {keys}")
        return preferred

    lowered = {key.lower(): key for key in keys}
    for candidate in candidates:
        for low, original in lowered.items():
            if candidate in low:
                return original
    raise KeyError(f"Could not infer key from {keys}; pass explicit key arguments.")


def pair_dir_name(template: str, source: str, target: str) -> str:
    return template.format(source=source, target=target, src=source, tgt=target)


def pair_score(
    npz_path: Path,
    target_dim: int,
    marginal_key: str | None,
    conditional_key: str | None,
) -> dict:
    import numpy as np

    data = load_npz(npz_path)
    keys = list(data.keys())
    marg_key = infer_key(keys, marginal_key, ("marg", "py", "p_y"))
    cond_key = infer_key(keys, conditional_key, ("cond", "pyx", "p_y_given_x"))

    marginal_nll = np.asarray(data[marg_key], dtype=float)
    conditional_nll = np.asarray(data[cond_key], dtype=float)
    if marginal_nll.shape[0] != conditional_nll.shape[0]:
        raise ValueError(f"NLL length mismatch in {npz_path}: {marginal_nll.shape} vs {conditional_nll.shape}")

    raw_is = float(marginal_nll.mean() - conditional_nll.mean())
    normalized_is = raw_is / float(target_dim)
    return {
        "raw_is": raw_is,
        "normalized_is": normalized_is,
        "n_samples": int(marginal_nll.shape[0]),
        "marginal_key": marg_key,
        "conditional_key": cond_key,
    }


def aggregate(values: list[float], method: str) -> float:
    import numpy as np

    if not values:
        return float("nan")
    if method == "median":
        return float(median(values))
    if method == "mean":
        return float(np.mean(values))
    if method == "trimmed_mean":
        if len(values) <= 2:
            return float(np.mean(values))
        arr = np.sort(np.asarray(values, dtype=float))
        return float(arr[1:-1].mean())
    raise ValueError(method)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate FLARE flow IS scores from per-pair NLL caches.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory containing pair subdirectories")
    parser.add_argument("--models", nargs="+", required=True, help="Model names in the candidate pool")
    parser.add_argument("--target-dims", type=Path, required=True, help="JSON mapping model name to embedding dimension")
    parser.add_argument("--pair-template", default="{source}_to_{target}_no_pca")
    parser.add_argument("--npz-name", default="per_example_nll.npz")
    parser.add_argument("--marginal-key")
    parser.add_argument("--conditional-key")
    parser.add_argument("--aggregation", choices=["median", "mean", "trimmed_mean"], default="median")
    parser.add_argument("--output", type=Path, default=Path("flare_is_scores.json"))
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_dims = load_json(args.target_dims)

    pair_results = []
    source_to_scores = {source: [] for source in args.models}

    for source in args.models:
        for target in args.models:
            if source == target:
                continue
            pair_dir = args.data_dir / pair_dir_name(args.pair_template, source, target)
            npz_path = pair_dir / args.npz_name
            if not npz_path.exists():
                if args.allow_missing:
                    continue
                raise FileNotFoundError(npz_path)
            if target not in target_dims:
                raise KeyError(f"Missing target dimension for {target}")

            score = pair_score(
                npz_path=npz_path,
                target_dim=int(target_dims[target]),
                marginal_key=args.marginal_key,
                conditional_key=args.conditional_key,
            )
            record = {
                "source_model": source,
                "target_model": target,
                "target_dim": int(target_dims[target]),
                "npz_path": str(npz_path),
                **score,
            }
            pair_results.append(record)
            source_to_scores[source].append(score["normalized_is"])

    model_scores = {
        source: aggregate(values, args.aggregation)
        for source, values in source_to_scores.items()
        if values or not args.allow_missing
    }

    output = {
        "method": "flare_flow_is",
        "data_dir": str(args.data_dir),
        "aggregation": args.aggregation,
        "models": args.models,
        "model_scores": model_scores,
        "pair_results": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    for model, score in sorted(model_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{model}\t{score:.6f}")


if __name__ == "__main__":
    main()
