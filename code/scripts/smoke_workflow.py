#!/usr/bin/env python3
"""Small CPU integration check. Scores are diagnostics, not research results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--with-embeddings', action='store_true',
                    help='Also test text encoding using MiniLM mean and CLS pooling')
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')

    def run(script, *arguments):
        subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                       env=env, check=True)

    if args.with_embeddings:
        texts = out / 'texts.txt'
        topics = ['science', 'finance', 'literature', 'music', 'history', 'software', 'sports', 'travel']
        texts.write_text('\n'.join(f'Example {i}: {topics[i % len(topics)]} research discusses {i*i+7} observations and topic {topics[(i//8)%len(topics)]}.' for i in range(64)), encoding='utf-8')
        for name, pooling in [('a', 'mean'), ('b', 'cls')]:
            run('scripts/generate_embeddings.py', '--input-file', texts, '--model', 'minilm',
                '--output-dir', out, '--output-name', name, '--pooling', pooling,
                '--batch-size', 8, '--max-length', 64, '--device', 'cpu', '--no-trust-remote-code')
    else:
        rng = np.random.default_rng(42)
        a = rng.normal(size=(64, 8)).astype(np.float32)
        b = (a @ rng.normal(size=(8, 6)) + .2 * rng.normal(size=(64, 6))).astype(np.float32)
        np.save(out / 'a.npy', a)
        np.save(out / 'b.npy', b)
    inputs = out / 'embeddings.json'
    inputs.write_text(json.dumps(dict(a='a.npy', b='b.npy')), encoding='utf-8')
    marginal = yaml.safe_load((ROOT / 'configs/marginal.yaml').read_text())
    marginal.update(layers=2, hidden_mult=.125 if args.with_embeddings else 2,
                    max_epochs=2, batch_size=128, spline_bins=4)
    conditional = yaml.safe_load((ROOT / 'configs/conditional.yaml').read_text())
    conditional.update(stage_epochs=[1, 1, 1], max_epochs=3, batch_size=16,
                       grad_accum_steps=1, adapter_rank=3, val_full_every=1)
    for name, config in [('marginal', marginal), ('conditional', conditional)]:
        (out / f'{name}.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    run('scripts/run_flow_pipeline.py', '--embeddings', inputs, '--output-dir', out / 'flows',
        '--marginal-config', out / 'marginal.yaml', '--conditional-config', out / 'conditional.yaml',
        '--val-size', .2, '--test-size', .2, '--device', 'cpu')
    scores = json.loads((out / 'flows/flare_is_scores.json').read_text())
    assert len(scores['pair_results']) == 2
    assert all(np.isfinite(v) for v in scores['model_scores'].values())
    for pair in scores['pair_results']:
        data = np.load(pair['npz_path'])
        expected = (data['marginal_nll'].astype(float).mean() - data['conditional_nll'].astype(float).mean()) / pair['target_dim']
        np.testing.assert_allclose(pair['normalized_is'], expected, rtol=1e-7, atol=1e-7)
        assert pair['n_samples'] == 13
    # Reject a target whose validation rows no longer match the marginal checkpoint.
    pair = out / 'flows/a_to_b_no_pca'
    y = np.load(pair / 'Y_val.npy')
    np.save(pair / 'Y_val.npy', y[::-1])
    rejection = subprocess.run([sys.executable, str(ROOT / 'scripts/score/export_per_example_nll.py'),
        '--data-dir', str(pair), '--marginal-model', str(out / 'flows/b_marginal/marginal_model.pth'),
        '--conditional-model', str(pair / 'cond_flow.pt'), '--split', 'test', '--device', 'cpu'],
        env=env, capture_output=True, text=True)
    np.save(pair / 'Y_val.npy', y)
    assert rejection.returncode != 0 and 'target data do not match' in rejection.stderr
    print('PASS: both directed pairs trained, NLL export and scores verified; mismatched targets rejected.')
    (out / 'SMOKE_RESULT.json').write_text(json.dumps(dict(passed=True, embedding_generation=args.with_embeddings,
        models='MiniLM mean/CLS variants' if args.with_embeddings else 'synthetic arrays',
        pairs=2, sample_count=64, held_out_samples=13,
        mismatch_rejected=True, cuda_visible_devices=env['CUDA_VISIBLE_DEVICES']), indent=2))


if __name__ == '__main__':
    main()
