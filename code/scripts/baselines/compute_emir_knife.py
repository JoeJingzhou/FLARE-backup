#!/usr/bin/env python3
"""Compute the EMIR/KNIFE baseline on embedding pairs.

This is intentionally separate from the FLARE normalizing-flow code. It is a
thin runner around the EMIR package's KNIFE estimator and expects embeddings
already saved as ``.npy`` or ``.npz`` files.

Example:
    python scripts/baselines/compute_emir_knife.py \
        --source-models gte_Qwen2_7B_instruct Linq_Embed_Mistral \
        --target-models all_MiniLM_L6_v2 stella_base_en_v2 \
        --embeddings-dir embeddings/almanach_arxiv_abstracts_2025 \
        --embedding-file abstract_embeddings.npy \
        --output-dir results/emir_knife \
        --device auto \
        --batch-size 256 \
        --emir-path /path/to/emir
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def add_emir_to_path(emir_path: Path | None) -> None:
    """Accept either the EMIR repo path or its parent directory."""
    if emir_path is None:
        return

    path = emir_path.expanduser().resolve()
    candidates = [path]
    if (path / "emir").is_dir():
        candidates.insert(0, path)
    if path.name == "emir" and (path / "__init__.py").exists():
        candidates.insert(0, path.parent)

    for candidate in candidates:
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


def load_knife_classes(emir_path: Path | None):
    add_emir_to_path(emir_path)
    try:
        module = importlib.import_module("emir.estimators.knife_estimator")
    except ImportError as exc:
        raise RuntimeError(
            "Could not import emir.estimators.knife_estimator. "
            "Pass --emir-path pointing to the EMIR repository, for example "
            "/path/to/emir."
        ) from exc
    return module.KNIFEEstimator, module.KNIFEArgs


def load_embedding(path: Path, normalize: bool):
    import numpy as np
    import torch

    if not path.exists() and path.with_suffix(".npz").exists():
        path = path.with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else ("embeddings" if "embeddings" in data else list(data.keys())[0])
        arr = data[key]
    else:
        arr = np.load(path)

    tensor = torch.as_tensor(arr).float()
    if tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    if normalize:
        tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
    return tensor


def run_knife(x, y, knife_estimator_cls, knife_args_cls, device: str, batch_size: int) -> dict[str, float]:
    knife_args = knife_args_cls(
        batch_size=batch_size,
        eval_batch_size=1024,
        n_epochs=100,
        n_epochs_marg=50,
        lr=1e-3,
        margin_lr=0.1,
        cond_modes=8,
        marg_modes=8,
        use_tanh=True,
        init_std=0.01,
        ff_residual_connection=False,
        ff_activation="relu",
        ff_layer_norm=True,
        ff_layers=2,
        ff_dim_hidden=0,
        device=device,
        stopping_criterion="max_epochs",
    )
    estimator = knife_estimator_cls(knife_args, x.shape[1], y.shape[1])
    mi, h_y, h_y_given_x = estimator.eval(x, y)
    return {
        "mi": float(mi),
        "h_y": float(h_y),
        "h_y_given_x": float(h_y_given_x),
    }


def model_embedding_path(embeddings_dir: Path, model: str, embedding_file: str) -> Path:
    return embeddings_dir / model / embedding_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute EMIR/KNIFE MI baseline for embedding pairs.")
    parser.add_argument("--source-models", "--model1", nargs="+", required=True)
    parser.add_argument("--target-models", "--model2", nargs="+", required=True)
    parser.add_argument("--embeddings-dir", "--embeddings_dir", type=Path, required=True)
    parser.add_argument("--embedding-file", "--embedding_file", default="sentences_embeddings.npy")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path("results/emir_knife"))
    parser.add_argument("--emir-path", "--emir_root", type=Path, help="Path to the EMIR repository or package parent")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    knife_estimator_cls, knife_args_cls = load_knife_classes(args.emir_path)
    import torch
    device = cuda if args.device == auto and torch.cuda.is_available() else (cpu if args.device == auto else args.device)

    dataset_name = args.embeddings_dir.name
    output_dir = args.output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "method": "emir_knife",
        "dataset": dataset_name,
        "embeddings_dir": str(args.embeddings_dir),
        "embedding_file": args.embedding_file,
        "device": args.device,
        "batch_size": args.batch_size,
        "normalize": args.normalize,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    results = []
    for source in args.source_models:
        source_path = model_embedding_path(args.embeddings_dir, source, args.embedding_file)
        print(f"Loading source {source}: {source_path}")
        x = load_embedding(source_path, args.normalize)

        for target in args.target_models:
            if source == target:
                print(f"Skipping self pair: {source}")
                continue

            pair_name = f"{source}_to_{target}"
            output_path = output_dir / f"{pair_name}.json"
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing result: {output_path}")
                results.append(json.loads(output_path.read_text(encoding="utf-8")))
                continue

            target_path = model_embedding_path(args.embeddings_dir, target, args.embedding_file)
            print(f"Loading target {target}: {target_path}")
            y = load_embedding(target_path, args.normalize)

            if x.shape[0] != y.shape[0]:
                raise ValueError(f"{pair_name} sample mismatch: {x.shape[0]} vs {y.shape[0]}")

            print(f"Computing EMIR/KNIFE for {pair_name}: X={tuple(x.shape)} Y={tuple(y.shape)}")
            metrics = run_knife(x, y, knife_estimator_cls, knife_args_cls, device, args.batch_size)
            result = {
                **metadata,
                "source_model": source,
                "target_model": target,
                **metrics,
            }
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)
            print(f"Saved {output_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
