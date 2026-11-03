import torch
import torch.nn as nn

from nflows import transforms, distributions
from nflows.flows import Flow
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.nn import nets as nflows_nets


class ScaledMLP(nn.Module):
    """
    """
    def __init__(self, in_shape, out_shape, hidden_sizes,
                 activation=nn.ReLU(), activate_output=False,
                 dropout_probability=0.0, output_scale=0.1):
        super().__init__()
        in_features  = in_shape[0]  if isinstance(in_shape,  (tuple,list)) else int(in_shape)
        out_features = out_shape[0] if isinstance(out_shape, (tuple,list)) else int(out_shape)
        hs = list(hidden_sizes) if isinstance(hidden_sizes, (list,tuple)) else [int(hidden_sizes)]
        layers = []
        dim_prev = in_features
        for h in hs:
            layers.append(nn.Linear(dim_prev, h))
            layers.append(activation if isinstance(activation, nn.Module) else nn.ReLU())
            if dropout_probability and dropout_probability > 0:
                layers.append(nn.Dropout(dropout_probability))
            dim_prev = h
        layers.append(nn.Linear(dim_prev, out_features))
        if activate_output:
            layers.append(activation if isinstance(activation, nn.Module) else nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.output_scale = output_scale

    def forward(self, inputs, context=None):
        return self.net(inputs) * self.output_scale


class SimpleCondMLP(nn.Module):
    """
    """
    def __init__(self, in_shape, out_shape, hidden_sizes, context_features=None,
                 activation=nn.ReLU(), activate_output=False,
                 dropout_probability=0.0, output_scale=0.1,
                 adapter_gain: float = 1e-2):
        super().__init__()
        in_features  = in_shape[0]  if isinstance(in_shape,  (tuple,list)) else int(in_shape)
        out_features = out_shape[0] if isinstance(out_shape, (tuple,list)) else int(out_shape)
        hs = list(hidden_sizes) if isinstance(hidden_sizes, (list,tuple)) else [int(hidden_sizes)]

        layers = []
        dim_prev = in_features
        for h in hs:
            layers.append(nn.Linear(dim_prev, h))
            layers.append(activation if isinstance(activation, nn.Module) else nn.ReLU())
            if dropout_probability and dropout_probability > 0:
                layers.append(nn.Dropout(dropout_probability))
            dim_prev = h
        layers.append(nn.Linear(dim_prev, out_features))
        if activate_output:
            layers.append(activation if isinstance(activation, nn.Module) else nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.output_scale = output_scale

        self.adapter = None
        self.adapter_gain = float(adapter_gain)
        if context_features is not None:
            self.adapter = nn.Linear(int(context_features), out_features)
            nn.init.zeros_(self.adapter.weight)
            nn.init.zeros_(self.adapter.bias)

    def forward(self, inputs, context=None):
        y = self.net(inputs)
        if self.adapter is not None and context is not None:
            y = y + self.adapter_gain * self.adapter(context)
        return y * self.output_scale


class ScaledResidualNet(nn.Module):
    """
    """
    def __init__(self, in_features, out_features, hidden_features, num_blocks=2,
                 activation=nn.ReLU(), dropout_probability=0.0, use_batch_norm=False,
                 context_features=None, output_scale=0.1):
        super().__init__()
        if isinstance(in_features, (tuple, list)):
            in_features = int(in_features[0])
        if isinstance(out_features, (tuple, list)):
            out_features = int(out_features[0])
        try:
            hidden_features = int(hidden_features)
        except Exception:
            pass
        self.net = nflows_nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            context_features=context_features,
        )
        self.output_scale = output_scale

    def forward(self, inputs, context=None):
        return self.net(inputs, context) * self.output_scale


def build_nsf(y_dim, hidden, layers, bins, use_actnorm=True, use_inv1x1=False):
    """
    """
    tfs = []
    even = True

    def net_factory(in_features, out_features):
        return ScaledMLP(
            in_shape=(in_features,),
            out_shape=(out_features,),
            hidden_sizes=[hidden, hidden],
            activation=nn.ReLU(),
            activate_output=False,
            dropout_probability=0.0,
            output_scale=0.1,
        )

    for _ in range(layers):
        mask = (torch.arange(y_dim) % 2 == (0 if even else 1)).float()

        def make_net_factory():
            return lambda in_f, out_f: net_factory(in_f, out_f)

        tfs.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net_factory(),
                tails='linear',
                tail_bound=3.0,
                num_bins=bins,
            )
        )
        if use_actnorm:
            tfs.append(transforms.ActNorm(features=y_dim))
        tfs.append(RandomPermutation(features=y_dim))
        even = not even

    return Flow(
        transform=transforms.CompositeTransform(tfs),
        distribution=distributions.StandardNormal(shape=[y_dim]),
    )


def build_cond_nsf(y_dim, x_dim, hidden, layers, bins, use_actnorm=True, use_inv1x1=False, dropout=0.0):
    """
    """
    tfs = []
    even = True

    def cond_net_factory(in_features, out_features, context_features):
        return ScaledResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden,
            num_blocks=2,
            activation=nn.ReLU(),
            dropout_probability=dropout,
            use_batch_norm=False,
            context_features=context_features,
            output_scale=0.1,
        )

    for _ in range(layers):
        mask = (torch.arange(y_dim) % 2 == (0 if even else 1)).float()

        def make_net_factory(context_dim):
            return lambda in_f, out_f: cond_net_factory(in_f, out_f, context_features=context_dim)

        tfs.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net_factory(x_dim),
                tails='linear',
                tail_bound=3.0,
                num_bins=bins,
            )
        )
        if use_actnorm:
            tfs.append(transforms.ActNorm(features=y_dim))
        tfs.append(RandomPermutation(features=y_dim))
        even = not even

    return Flow(
        transform=transforms.CompositeTransform(tfs),
        distribution=distributions.StandardNormal(shape=[y_dim]),
    )


def build_cond_nsf_same_structure(y_dim, x_dim, hidden, layers, bins, use_actnorm=True, use_inv1x1=False, dropout=0.0):
    """
    """
    tfs = []
    even = True

    def cond_net_factory(in_features, out_features, context_features):
        return SimpleCondMLP(
            in_shape=in_features,
            out_shape=out_features,
            hidden_sizes=[hidden] * 2,
            context_features=context_features,
            activation=nn.ReLU(),
            activate_output=False,
            dropout_probability=dropout,
            output_scale=0.1,
            adapter_gain=1e-2,
        )

    for _ in range(layers):
        mask = (torch.arange(y_dim) % 2 == (0 if even else 1)).float()

        def make_net_factory(context_dim):
            return lambda in_f, out_f: cond_net_factory(in_f, out_f, context_features=context_dim)

        tfs.append(
            PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=make_net_factory(x_dim),
                tails='linear',
                tail_bound=3.0,
                num_bins=bins,
                min_bin_width=1e-2,
                min_bin_height=1e-2,
                min_derivative=1e-2,
            )
        )
        if use_actnorm:
            tfs.append(transforms.ActNorm(features=y_dim))
        tfs.append(RandomPermutation(features=y_dim))
        even = not even

    return Flow(
        transform=transforms.CompositeTransform(tfs),
        distribution=distributions.StandardNormal(shape=[y_dim]),
    )
