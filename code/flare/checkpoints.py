"""Checkpoint interfaces shared by training and scoring (release flow classes)."""
from pathlib import Path
import hashlib

import torch

from .flows.marginal import MarginalFlow
from .flows.conditional import ConditionalFlow


def target_data_signature(data_dir):
    """Detect different target transforms, row orders or split assignments."""
    result = {}
    for name in ("Y_train.npy", "Y_val.npy"):
        digest = hashlib.sha256()
        with (Path(data_dir) / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def marginal_kwargs(config, y_dim):
    return dict(y_dim=int(y_dim),
                hidden=max(1, int(min(2048, float(config['hidden_mult']) * y_dim))),
                layers=int(config['layers']), bins=int(config['spline_bins']),
                use_actnorm=bool(config.get('use_actnorm', True)),
                use_inv1x1=bool(config.get('use_inv1x1', False)))


def load_flow(path, kind, *, y_dim, x_dim=None, device='cpu'):
    """Load tensor/primitive checkpoints strictly; support original release metadata."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if kind not in ('marginal', 'conditional'):
        raise ValueError(f'Unknown flow kind: {kind}')
    config = checkpoint['cfg' if kind == 'marginal' else 'config']
    kwargs = checkpoint.get('model_kwargs')
    if kwargs is None:
        kwargs = marginal_kwargs(config, y_dim)
        if kind == 'conditional':
            kwargs.update(x_dim=int(x_dim), dropout=float(config.get('dropout', 0.0)),
                          output_scale=float(config.get('output_scale', 0.1)),
                          adapter_rank=int(config.get('adapter_rank', 64)),
                          adapter_scale=float(config.get('adapter_scale', 2e-3)))
    else:
        kwargs = dict(kwargs)
    if kwargs['y_dim'] != y_dim or (kind == 'conditional' and kwargs['x_dim'] != x_dim):
        raise ValueError(f'Checkpoint dimensions do not match data: {kwargs}')
    cls = MarginalFlow if kind == 'marginal' else ConditionalFlow
    model = cls(**kwargs).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint, kwargs
