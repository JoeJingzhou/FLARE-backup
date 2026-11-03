import torch
import torch.nn as nn

from nflows import transforms, distributions
from nflows.flows import Flow
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform


class ScaledMLP(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_sizes,
                 activation=nn.ReLU(), activate_output=False,
                 dropout_probability=0.0, output_scale=0.1):
        super().__init__()
        in_features  = in_shape[0]  if isinstance(in_shape,  (tuple, list)) else int(in_shape)
        out_features = out_shape[0] if isinstance(out_shape, (tuple, list)) else int(out_shape)
        hs = list(hidden_sizes) if isinstance(hidden_sizes, (list, tuple)) else [int(hidden_sizes)]

        layers, dim_prev = [], in_features
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
        self.output_scale = float(output_scale)

    def forward(self, inputs, context=None):
        return self.net(inputs) * self.output_scale


class LoRAAdapter(nn.Module):
    def __init__(self, context_features: int, out_features: int,
                 rank: int = 64, scale: float = 2e-3, bias: bool = True):
        super().__init__()
        self.rank = int(rank)
        self.scale = float(scale)

        self.A = nn.Linear(context_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, out_features, bias=bias)

        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)
        if self.B.bias is not None:
            nn.init.zeros_(self.B.bias)

    def forward(self, context):
        if context is None:
            return 0.0
        z = self.A(context)                 # [B, r]
        y = self.B(z)                       # [B, out_features]
        return y * self.scale


class MLPWithLoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes,
                 context_features, adapter_rank=64, adapter_scale=2e-3,
                 activation=nn.ReLU(), dropout_probability=0.0, output_scale=0.1):
        super().__init__()
        self.base = ScaledMLP(
            in_shape=(in_features,),
            out_shape=(out_features,),
            hidden_sizes=hidden_sizes,
            activation=activation,
            activate_output=False,
            dropout_probability=dropout_probability,
            output_scale=output_scale,
        )
        self.adapter = LoRAAdapter(
            context_features=context_features,
            out_features=out_features,
            rank=int(adapter_rank),
            scale=float(adapter_scale),
            bias=True,
        )

    def forward(self, inputs, context=None):
        base_out = self.base(inputs)                    # [B, out_features]
        ada_out  = self.adapter(context) if context is not None else 0.0
        return base_out + ada_out


class ConditionalFlow(nn.Module):
    """
      output = base_transform(inputs) + scale * B(A(context))
    """
    def __init__(self, y_dim, x_dim, hidden, layers, bins,
                 use_actnorm=True, use_inv1x1=False, dropout=0.0,
                 output_scale=0.1, adapter_rank=64, adapter_scale=2e-3):
        super().__init__()
        self.y_dim = int(y_dim)
        self.x_dim = int(x_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.bins = int(bins)
        self.use_actnorm = bool(use_actnorm)
        self.use_inv1x1 = bool(use_inv1x1)
        self.dropout = float(dropout)
        self.output_scale = float(output_scale)
        self.adapter_rank = int(adapter_rank)
        self.adapter_scale = float(adapter_scale)

        tfs = []
        even = True

        def make_net_factory(context_dim):
            def factory(in_features, out_features):
                return MLPWithLoRAAdapter(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_sizes=[self.hidden, self.hidden],
                    context_features=context_dim,
                    adapter_rank=self.adapter_rank,
                    adapter_scale=self.adapter_scale,
                    activation=nn.ReLU(),
                    dropout_probability=self.dropout,
                    output_scale=self.output_scale,
                )
            return factory

        for _ in range(self.layers):
            mask = (torch.arange(self.y_dim) % 2 == (0 if even else 1)).float()
            tfs.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=mask,
                    transform_net_create_fn=make_net_factory(self.x_dim),
                    tails='linear',
                    tail_bound=3.0,
                    num_bins=self.bins,
                )
            )
            if self.use_actnorm:
                tfs.append(transforms.ActNorm(features=self.y_dim))
            tfs.append(RandomPermutation(features=self.y_dim))
            even = not even

        self.flow = Flow(
            transform=transforms.CompositeTransform(tfs),
            distribution=distributions.StandardNormal(shape=[self.y_dim]),
        )

    def log_prob(self, y, context):
        return self.flow.log_prob(y, context=context)

    def sample(self, num_samples, context):
        return self.flow.sample(num_samples, context=context)
