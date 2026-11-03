#!/usr/bin/env python3
"""Export paired held-out NLLs from release flow checkpoints for FLARE scoring."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flare.checkpoints import load_flow, target_data_signature


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--marginal-model', type=Path, required=True)
    ap.add_argument('--conditional-model', type=Path, required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.batch_size <= 0:
        ap.error('--batch-size must be positive')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    x = np.load(args.data_dir / f'X_{args.split}.npy', mmap_mode='r')
    y = np.load(args.data_dir / f'Y_{args.split}.npy', mmap_mode='r')
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(y):
        raise ValueError('Expected nonempty, aligned two-dimensional X/Y arrays')
    marginal, mck, _ = load_flow(args.marginal_model, 'marginal', y_dim=y.shape[1], device=device)
    conditional, cck, _ = load_flow(args.conditional_model, 'conditional',
                                    y_dim=y.shape[1], x_dim=x.shape[1], device=device)
    signature = target_data_signature(args.data_dir)
    for name, checkpoint in [('marginal', mck), ('conditional', cck)]:
        if checkpoint.get('target_data_signature') != signature:
            raise ValueError(f'{name} checkpoint target data do not match; use this release to retrain')
    marginal_nll, conditional_nll = [], []
    with torch.no_grad():
        for start in range(0, len(y), args.batch_size):
            xb = torch.from_numpy(np.array(x[start:start + args.batch_size], dtype=np.float32)).to(device)
            yb = torch.from_numpy(np.array(y[start:start + args.batch_size], dtype=np.float32)).to(device)
            marginal_nll.append((-marginal.log_prob(yb)).cpu().numpy())
            conditional_nll.append((-conditional.log_prob(yb, context=xb)).cpu().numpy())
    marginal_nll = np.concatenate(marginal_nll)
    conditional_nll = np.concatenate(conditional_nll)
    if not np.isfinite(marginal_nll).all() or not np.isfinite(conditional_nll).all():
        raise ValueError('Non-finite NLLs; refusing to produce a scoring cache')
    output = args.output or args.data_dir / 'per_example_nll.npz'
    if output.suffix != '.npz':
        raise ValueError('--output must end with .npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, marginal_nll=marginal_nll, conditional_nll=conditional_nll)
    metadata = dict(split=args.split, n_samples=len(y), x_dim=x.shape[1], y_dim=y.shape[1],
                    data_dir=str(args.data_dir.resolve()),
                    marginal_model=str(args.marginal_model.resolve()),
                    conditional_model=str(args.conditional_model.resolve()),
                    target_data_signature=signature,
                    raw_is=float(marginal_nll.mean() - conditional_nll.mean()))
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {output}: {len(y)} aligned {args.split} NLL pairs')


if __name__ == '__main__':
    main()
