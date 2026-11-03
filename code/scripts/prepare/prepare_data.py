import numpy as np, os, argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split
from flare.preprocess import fit_scalers
from tqdm import tqdm
import joblib

def _load_scalers(scalers_dir: str):
    xs_path = os.path.join(scalers_dir, 'x_scaler.joblib')
    ys_path = os.path.join(scalers_dir, 'y_scaler.joblib')
    pca_path = os.path.join(scalers_dir, 'y_pca.joblib')

    xs = joblib.load(xs_path) if os.path.exists(xs_path) else None
    ys = joblib.load(ys_path) if os.path.exists(ys_path) else None
    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None
    return xs, ys, pca

def _transform_split_with_progress(x, y, idx, scalers_dir: str, chunk_size: int = 8192):
    """
    """
    xs, ys, pca = _load_scalers(scalers_dir)

    N = len(idx)
    if y is None:
        raise ValueError("Invalid input format")
    y_dim_out = (pca.n_components_ if pca is not None else y.shape[1])

    X_out = None
    if x is not None and xs is not None:
        X_out = np.empty((N, x.shape[1]), dtype=np.float32)
    Y_out = np.empty((N, y_dim_out), dtype=np.float32)

    for start in tqdm(range(0, N, chunk_size), desc="Transforming", unit="chunk"):
        end = min(start + chunk_size, N)
        batch_idx = idx[start:end]

        y_batch = y[batch_idx]
        y_std = ys.transform(y_batch) if ys is not None else y_batch
        y_t = pca.transform(y_std) if pca is not None else y_std
        Y_out[start:end] = y_t.astype(np.float32, copy=False)

        if X_out is not None:
            x_batch = x[batch_idx]
            x_t = xs.transform(x_batch)
            X_out[start:end] = x_t.astype(np.float32, copy=False)

    return X_out, Y_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x_path', type=str, default=None)
    ap.add_argument('--y_path', type=str, required=True)
    ap.add_argument('--output_dir', type=str, default='data')
    ap.add_argument('--test_size', type=float, default=0.1)
    ap.add_argument('--val_size', type=float, default=0.1)
    ap.add_argument('--pca_keep_var', type=float, default=0.98)
    ap.add_argument('--x_pca_keep_var', type=float, default=None)
    ap.add_argument('--chunk_size', type=int, default=8192, help="See README for details")
    args = ap.parse_args()
    if args.val_size <= 0 or args.test_size < 0 or args.val_size + args.test_size >= 1:
        ap.error('Require val_size > 0, test_size >= 0 and their sum < 1')
    if args.chunk_size <= 0:
        ap.error('chunk_size must be positive')

    print("Loading embeddings...")
    y = np.load(args.y_path, mmap_mode='r')
    x = np.load(args.x_path, mmap_mode='r') if args.x_path is not None else None
    print(f"y shape = {y.shape}")
    if x is not None:
        print(f"x shape = {x.shape}")
        if len(x) != len(y):
            raise ValueError('X and Y must have the same number of samples.')
    for name, array in [('Y', y), ('X', x)]:
        if array is not None and (array.ndim != 2 or not len(array) or array.shape[1] < 2):
            raise ValueError(f'{name} must be a nonempty 2D embedding matrix with at least two features')
    if (Path(args.output_dir) / 'scalers/y_pca.joblib').exists():
        raise ValueError('This release uses standardization only; use a clean output directory without stale PCA')

    os.makedirs(args.output_dir, exist_ok=True)

    print("Splitting indices (train/val/test)...")
    idx_all = np.arange(len(y))
    
    if args.test_size == 0.0:
        idx_train, idx_val = train_test_split(
            idx_all, test_size=args.val_size, random_state=42, shuffle=True
        )
        idx_test = np.array([])
    else:
        idx_train, idx_tmp = train_test_split(
            idx_all, test_size=args.val_size + args.test_size, random_state=42, shuffle=True
        )
        rel = args.test_size / (args.val_size + args.test_size) if (args.val_size + args.test_size) > 0 else 0.5
        idx_val, idx_test = train_test_split(idx_tmp, test_size=rel, random_state=42, shuffle=True)

    print("Fitting scalers on TRAIN only (standardization; this release does not apply PCA)...")
    y_train_fit = y[idx_train]
    x_train_fit = x[idx_train] if x is not None else None
    scalers_dir = os.path.join(args.output_dir, 'scalers')
    fit_scalers(x=x_train_fit, y=y_train_fit, pca_keep_var=args.pca_keep_var, out_dir=scalers_dir)

    splits_dir = os.path.join(args.output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    np.save(os.path.join(splits_dir, 'idx_train.npy'), idx_train)
    np.save(os.path.join(splits_dir, 'idx_val.npy'), idx_val)
    np.save(os.path.join(splits_dir, 'idx_test.npy'), idx_test)

    splits_to_save = [('train', idx_train), ('val', idx_val)]
    if len(idx_test) > 0:
        splits_to_save.append(('test', idx_test))
    
    for name, idx in splits_to_save:
        print(f"Transforming & saving {name} split...")
        X_t, Y_t = _transform_split_with_progress(
            x=x, y=y, idx=idx, scalers_dir=scalers_dir, chunk_size=args.chunk_size
        )
        if X_t is not None:
            np.save(os.path.join(args.output_dir, f'X_{name}.npy'), X_t)
        np.save(os.path.join(args.output_dir, f'Y_{name}.npy'), Y_t)
        print(f"Saved: {('X_'+name+'.npy, ' if X_t is not None else '')}Y_{name}.npy")

    meta = {
        "x_path": args.x_path,
        "y_path": args.y_path,
        "pca_keep_var": args.pca_keep_var,
        "preprocessing": "StandardScaler fitted on train only; no PCA",
        "legacy_pca_flags_effective": False,
        "split_seed": 42,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "chunk_size": args.chunk_size,
        "n_samples": int(len(y)),
    }
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Data prepared.')

if __name__ == '__main__':
    main()
