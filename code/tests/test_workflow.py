import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flare.checkpoints import load_flow, marginal_kwargs, target_data_signature
from flare.flows.marginal import MarginalFlow
from flare.flows.conditional import ConditionalFlow
from flare.utils import EMA

spec = importlib.util.spec_from_file_location('train_conditional', ROOT / 'scripts/train/train_conditional.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class WorkflowTests(unittest.TestCase):
    def test_fractional_hidden_multiplier_is_an_integer_width(self):
        kwargs = marginal_kwargs(dict(hidden_mult=.35, layers=1, spline_bins=4), 8)
        self.assertEqual(kwargs['hidden'], 2)
        MarginalFlow(**kwargs)

    def test_ema_preserves_permutations_and_boolean_buffers(self):
        model = MarginalFlow(8, hidden=8, layers=2, bins=4)
        ema = EMA(model)
        before = {k: v.clone() for k, v in model.state_dict().items() if not v.is_floating_point()}
        for _ in range(20):
            ema.update(model)
        ema.load_shadow(model)
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(ema.shadow[name].dtype, value.dtype)

    def test_transfer_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        kwargs = dict(y_dim=8, hidden=12, layers=2, bins=4,
                      use_actnorm=True, use_inv1x1=False)
        marginal = MarginalFlow(**kwargs)
        y, x = torch.randn(16, 8), torch.randn(16, 6)
        marginal.log_prob(y)
        conditional_kwargs = dict(kwargs, x_dim=6, dropout=0., output_scale=.1,
                                  adapter_rank=3, adapter_scale=.004)
        conditional = ConditionalFlow(**conditional_kwargs)
        training.copy_shared_weights(marginal, conditional)
        marginal.eval(); conditional.eval()
        torch.testing.assert_close(conditional.log_prob(y, x), marginal.log_prob(y), rtol=1e-6, atol=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cond.pt'
            torch.save(dict(state_dict=conditional.state_dict(), model_kwargs=conditional_kwargs,
                            config=dict(layers=99, hidden_mult=.35, spline_bins=4)), path)
            restored, _, _ = load_flow(path, 'conditional', y_dim=8, x_dim=6)
            torch.testing.assert_close(restored.log_prob(y, x), conditional.log_prob(y, x))
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                load_flow(path, 'conditional', y_dim=8, x_dim=7)

    def test_schedule_and_total_epoch_cap(self):
        trainer = training.ThreeStageTrainer(dict(stage_epochs=[2, 3, 4], max_epochs=4), torch.device('cpu'))
        self.assertEqual([s['epochs'] for s in trainer.stages], [2, 2])
        self.assertFalse(trainer.amp)
        with self.assertRaises(ValueError):
            training.ThreeStageTrainer(dict(stage_epochs=[0, 0, 0]), torch.device('cpu'))

    def test_signature_detects_target_row_order_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            np.save(p / 'Y_train.npy', np.arange(24).reshape(4, 6))
            np.save(p / 'Y_val.npy', np.arange(12).reshape(2, 6))
            before = target_data_signature(p)
            np.save(p / 'Y_val.npy', np.arange(12).reshape(2, 6)[::-1])
            self.assertNotEqual(before, target_data_signature(p))


if __name__ == '__main__':
    unittest.main()
