#!/usr/bin/env python3
"""Prepare, train, export NLLs, and score every directed embedding pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--embeddings', type=Path, required=True,
                    help='JSON mapping model names to .npy paths; relative paths resolve beside JSON')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--marginal-config', type=Path, default=ROOT / 'configs/marginal.yaml')
    ap.add_argument('--conditional-config', type=Path, default=ROOT / 'configs/conditional.yaml')
    ap.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    ap.add_argument('--val-size', type=float, default=0.1)
    ap.add_argument('--test-size', type=float, default=0.1)
    ap.add_argument('--score-split', choices=['val', 'test'], default='test')
    args = ap.parse_args()
    if args.score_split == 'test' and args.test_size <= 0:
        ap.error('Test scoring requires --test-size > 0; use --score-split val otherwise')
    if args.val_size <= 0 or args.test_size < 0 or args.val_size + args.test_size >= 1:
        ap.error('Require a positive validation split and val_size + test_size < 1')
    inputs = json.loads(args.embeddings.read_text(encoding='utf-8'))
    if len(inputs) < 2:
        ap.error('At least two embedding models are required')
    paths, dims, rows, digests = {}, {}, set(), {}
    for name, value in inputs.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            ap.error(f'Use a simple filesystem-safe model name: {name}')
        path = Path(value)
        path = (args.embeddings.resolve().parent / path).resolve() if not path.is_absolute() else path.resolve()
        matrix = np.load(path, mmap_mode='r')
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            ap.error(f'{name} must be a 2D embedding matrix with at least two features')
        paths[name], dims[name] = path, matrix.shape[1]
        rows.add(len(matrix))
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(block)
        digests[name] = digest.hexdigest()
    if len(rows) != 1:
        ap.error('All embedding matrices must have the same rows in the same sample order')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if args.device == 'cpu':
        environment['CUDA_VISIBLE_DEVICES'] = ''
    manifest = dict(embeddings={k: str(v) for k, v in paths.items()}, embedding_sha256=digests,
                    target_dims=dims, score_split=args.score_split, commands=[], completed=False,
                    marginal_config=args.marginal_config.read_text(),
                    conditional_config=args.conditional_config.read_text())

    def run(script, *arguments):
        command = [sys.executable, str(ROOT / script), *map(str, arguments)]
        manifest['commands'].append(command)
        (out / 'workflow_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('+', ' '.join(command), flush=True)
        subprocess.run(command, env=environment, check=True)

    def prepare(target, directory, source=None):
        extra = ['--x_path', paths[source]] if source else []
        run('scripts/prepare/prepare_data.py', '--y_path', paths[target], '--output_dir', directory,
            '--val_size', args.val_size, '--test_size', args.test_size, *extra)

    for target in inputs:
        marginal_dir = out / f'{target}_marginal'
        prepare(target, marginal_dir)
        marginal = marginal_dir / 'marginal_model.pth'
        run('scripts/train/train_marginal.py', '--config', args.marginal_config,
            '--data_dir', marginal_dir, '--out', marginal)
        for source in inputs:
            if source == target:
                continue
            pair = out / f'{source}_to_{target}_no_pca'
            prepare(target, pair, source)
            run('scripts/train/train_conditional.py', '--config', args.conditional_config,
                '--data_dir', pair, '--marginal_model', marginal, '--out', pair / 'cond_flow.pt')
            run('scripts/score/export_per_example_nll.py', '--data-dir', pair,
                '--marginal-model', marginal, '--conditional-model', pair / 'cond_flow.pt',
                '--split', args.score_split, '--device', args.device)
    dimensions = out / 'model_dims.json'
    dimensions.write_text(json.dumps(dims, indent=2), encoding='utf-8')
    run('scripts/score/compute_flare_is.py', '--data-dir', out, '--models', *inputs,
        '--target-dims', dimensions, '--output', out / 'flare_is_scores.json')
    manifest['completed'] = True
    (out / 'workflow_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
