#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm.auto import trange

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flare.flows.marginal import MarginalFlow
from flare.flows.conditional import ConditionalFlow
from flare.checkpoints import load_flow, target_data_signature


def copy_shared_weights(marginal, conditional):
    """
    """
    print("Processing...")
    marg_sd = marginal.flow.state_dict()
    cond_sd = conditional.flow.state_dict()

    copied = 0
    skipped = 0

    for name, m in marg_sd.items():
        if name in cond_sd and cond_sd[name].shape == m.shape:
            cond_sd[name].copy_(m)
            copied += 1
        else:
            skipped += 1

    alias_copied = 0
    for name, m in marg_sd.items():
        if ".transform_net." in name:
            alias = name.replace(".transform_net.", ".transform_net.base.")
            if alias in cond_sd and cond_sd[alias].shape == m.shape:
                cond_sd[alias].copy_(m)
                alias_copied += 1

    if copied + alias_copied != len(marg_sd):
        raise ValueError('Conditional backbone does not match the marginal checkpoint')
    conditional.flow.load_state_dict(cond_sd, strict=True)
    print(f"Copied all {copied + alias_copied} marginal state entries")


def zero_init_adapter_B_only(model):
    """Release pipeline helper."""
    print("Processing...")
    zeroed = 0
    for name, p in model.named_parameters():
        if "adapter.B.weight" in name:
            with torch.no_grad():
                p.zero_()
            zeroed += 1
            print(f"  zero: {name} {tuple(p.shape)}")
    print("Processing...")


def eval_nll_marg(marginal, y, batch_size=8192):
    marginal.eval()
    n = y.size(0)
    tot = 0.0
    for i in range(0, n, batch_size):
        yp = y[i:i+batch_size]
        logp = marginal.flow.log_prob(yp)
        tot += (-logp).sum().item()
    return tot / n


@torch.no_grad()
def eval_nll_cond(cond, x, y, batch_size=8192, use_amp=True):
    cond.eval()
    n = y.size(0)
    tot = 0.0
    use_amp = use_amp and (x.device.type == "cuda")
    for i in range(0, n, batch_size):
        xp = x[i:i+batch_size]
        yp = y[i:i+batch_size]
        with autocast(device_type='cuda', enabled=use_amp):
            logp = cond.flow.log_prob(yp, context=xp)
        tot += (-logp).sum().item()
    return tot / n


@torch.no_grad()
def eval_nll_cond_indices(cond, x, y, idx, batch_size=8192, use_amp=True):
    """Release pipeline helper."""
    cond.eval()
    xsub = x.index_select(0, idx)
    ysub = y.index_select(0, idx)
    return eval_nll_cond(cond, xsub, ysub, batch_size=batch_size, use_amp=use_amp)


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0, stage_name="Unknown"):
        self.patience = patience
        self.min_delta = min_delta
        self.stage_name = stage_name
        self.best = None
        self.bad = 0

    def step(self, val):
        if self.best is None or (val < self.best - self.min_delta):
            self.best = val
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


class ThreeStageTrainer:
    def __init__(self, cfg, device, base_lr=1e-3, min_lr=1e-5):
        self.cfg = cfg
        self.device = device
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.amp = bool(cfg.get('use_amp', False))
        self.scaler = GradScaler('cuda', enabled=(self.amp and device.type == "cuda"))

        self.val_subset_ratio = float(cfg.get('val_subset_ratio', 0.1))
        self.val_subset_min = int(cfg.get('val_subset_min', 1024))
        self.val_subset_size = int(cfg.get('val_subset_size', 0))
        self.val_full_every = int(cfg.get('val_full_every', 10))

        self.stages = [
            dict(name="Stage1", epochs=60, unfreeze_ratio=0.0, lr_factor=1.0, patience=8, min_delta=1e-5),
            dict(name="Stage2", epochs=60, unfreeze_ratio=0.5, lr_factor=1.0, patience=8, min_delta=1e-5),
            dict(name="Stage3", epochs=220, unfreeze_ratio=1.0, lr_factor=1.0, patience=12, min_delta=1e-5),
        ]
        epochs = cfg.get('stage_epochs', [60, 60, 220])
        if len(epochs) != 3 or any(int(n) != n or n < 0 for n in epochs):
            raise ValueError('stage_epochs must contain three nonnegative integers')
        budget = int(cfg.get('max_epochs', sum(epochs)))
        if budget <= 0 or sum(epochs) <= 0:
            raise ValueError('The training epoch budget must be positive')
        for stage, count in zip(self.stages, epochs):
            stage['epochs'] = min(int(count), budget)
            budget -= stage['epochs']
        self.stages = [stage for stage in self.stages if stage['epochs'] > 0]

    def build_val_subset_idx(self, n_val, seed=43, device='cpu'):
        if self.val_subset_size > 0:
            k = min(n_val, self.val_subset_size)
        else:
            k = max(self.val_subset_min, int(n_val * self.val_subset_ratio))
            k = min(k, n_val)
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        idx_cpu = torch.randperm(n_val, generator=g)[:k]
        return idx_cpu.to(device)

    def unfreeze_partial_backbone(self, model, ratio, skip_actnorm=True):
        """Release pipeline helper."""
        flow_params = []
        for n, p in model.flow.named_parameters():
            if 'adapter' in n:
                continue
            if skip_actnorm and 'actnorm' in n.lower():
                continue
            flow_params.append((n, p))
        num_unfreeze = int(len(flow_params) * ratio)
        unfrozen = 0
        for i, (n, p) in enumerate(flow_params):
            if i >= len(flow_params) - num_unfreeze and not p.requires_grad:
                p.requires_grad = True
                unfrozen += 1
        print("Processing...")

    def build_optim(self, model, lr_adapter=1e-3, lr_backbone=2e-4):
        """Release pipeline helper."""
        groups = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'adapter' in n:
                groups.append({'params': [p], 'lr': lr_adapter})
            else:
                groups.append({'params': [p], 'lr': lr_backbone})
        print("Processing...")
        opt = torch.optim.AdamW(groups, weight_decay=float(self.cfg.get('weight_decay', 0.0)))
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.8, patience=2, min_lr=self.min_lr
        )
        return opt, sch

    def train_one_epoch(self, cond, marginal, opt, x, y, batch_size, grad_accum_steps):
        cond.train()
        n = x.size(0)
        idx = torch.randperm(n, device=x.device)
        xb = x.index_select(0, idx)
        yb = y.index_select(0, idx)

        scaler = self.scaler
        use_amp = self.amp and (x.device.type == "cuda")

        loss_sum, steps = 0.0, 0
        for i in range(0, n, batch_size * grad_accum_steps):
            opt.zero_grad(set_to_none=True)
            for j in range(grad_accum_steps):
                s, e = i + j * batch_size, i + (j + 1) * batch_size
                if s >= n:
                    break
                xj = xb[s:e]
                yj = yb[s:e]
                with autocast(device_type='cuda', enabled=use_amp):
                    logp_c = cond.flow.log_prob(yj, context=xj)
                    nll_c = -logp_c.mean()
                    loss = nll_c / grad_accum_steps
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_sum += loss.item()
                steps += 1
            if use_amp:
                scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in cond.parameters() if p.requires_grad], max_norm=1.0)
            if use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
        return loss_sum / max(1, steps)

    def fit(self, marginal, conditional, x_train, y_train, x_val, y_val, out_path):
        eval_bs = int(self.cfg.get('eval_bs', 8192))

        val_idx = self.build_val_subset_idx(n_val=y_val.size(0),
                                            seed=int(self.cfg.get('seed', 66)),
                                            device=x_val.device)

        val_nll_marg = eval_nll_marg(marginal, y_val, batch_size=eval_bs)
        print("Processing...")
        
        y_val_sub = y_val.index_select(0, val_idx)
        val_nll_marg_sub = eval_nll_marg(marginal, y_val_sub, batch_size=eval_bs)
        print("Processing...")

        with torch.no_grad():
            init_cond_full = eval_nll_cond(conditional, x_val, y_val, batch_size=eval_bs, use_amp=False)
            init_cond_sub = eval_nll_cond_indices(conditional, x_val, y_val, val_idx, batch_size=eval_bs, use_amp=False)
        if abs(init_cond_full - val_nll_marg) > 1e-4:
            raise ValueError('Transferred conditional flow must initially equal the marginal flow')
        print("Processing...")
        if abs(init_cond_full - val_nll_marg) <= 1.0:
            print("Processing...")
        else:
            print("Processing...")
        
        if abs(init_cond_sub - val_nll_marg_sub) <= 1.0:
            print("Processing...")
        else:
            print("Processing...")

        best_overall = dict(nll=init_cond_full,
                            state={k: v.detach().cpu().clone() for k, v in conditional.state_dict().items()},
                            epoch=0, delta=init_cond_full - val_nll_marg, stage="initial")
        total_epoch = 0

        for si, st in enumerate(self.stages):
            is_last_stage = (si == len(self.stages) - 1)
            print(f"\n=== {st['name']} ===")

            if si > 0:
                conditional.load_state_dict(best_overall['state'], strict=True)

            skip_act = (st['name'] != "Stage3")
            if st['unfreeze_ratio'] > 0.0:
                self.unfreeze_partial_backbone(conditional, st['unfreeze_ratio'], skip_actnorm=skip_act)

            if st['name'] == "Stage1":
                opt, sch = self.build_optim(conditional, lr_adapter=self.base_lr, lr_backbone=0.0)
            elif st['name'] == "Stage2":
                opt, sch = self.build_optim(conditional, lr_adapter=self.base_lr,
                                            lr_backbone=self.base_lr*0.2)
            else:
                opt, sch = self.build_optim(conditional, lr_adapter=self.base_lr,
                                            lr_backbone=self.base_lr*0.1)

            stopper = EarlyStopper(patience=st['patience'], min_delta=st['min_delta'], stage_name=st['name'])
            rollback_count = 0
            best_val_seen = float('inf')
            advance_stage = False
            stop_training = False

            best_stage = dict(nll=float('inf'), state=None, epoch=0, delta=float('inf'))

            pbar = trange(st['epochs'], desc=st['name'], leave=False)
            for ep in pbar:
                total_epoch += 1

                tr_loss = self.train_one_epoch(
                    conditional, marginal, opt, x_train, y_train,
                    batch_size=int(self.cfg.get('batch_size', 256)),
                    grad_accum_steps=int(self.cfg.get('grad_accum_steps', 1))
                )

                val_nll_cond_sub = eval_nll_cond_indices(
                    conditional, x_val, y_val, val_idx, batch_size=eval_bs, use_amp=self.amp
                )
                delta_sub = val_nll_cond_sub - val_nll_marg_sub

                if val_nll_cond_sub < best_val_seen - st['min_delta']:
                    best_val_seen = val_nll_cond_sub
                    rollback_count = 0

                old_lr = min(pg['lr'] for pg in opt.param_groups)
                sch.step(val_nll_cond_sub)
                new_lr = min(pg['lr'] for pg in opt.param_groups)
                
                lr_adapter_cur = max(pg['lr'] for pg in opt.param_groups)
                
                if lr_adapter_cur < float(self.cfg.get('min_adapter_lr', 0.004)):
                    if is_last_stage:
                        print("Processing...")
                        stop_training = True
                    else:
                        print("Processing...")
                        advance_stage = True
                    val_nll_cond_full = eval_nll_cond(conditional, x_val, y_val, batch_size=eval_bs, use_amp=self.amp)
                    delta_full = val_nll_cond_full - val_nll_marg
                    print(f"[full-eval@lr-threshold] val(full)={val_nll_cond_full:.6f} Δ(full)={delta_full:.6f}")
                    break

                if new_lr < old_lr:
                    rollback_count += 1
                    print(f"[lr-decay] minLR {old_lr:.6f} -> {new_lr:.6f} | rollback_count={rollback_count}")
                    if rollback_count >= 3:
                        if is_last_stage:
                            print("Processing...")
                            stop_training = True
                        else:
                            print("Processing...")
                            advance_stage = True
                        val_nll_cond_full = eval_nll_cond(conditional, x_val, y_val, batch_size=eval_bs, use_amp=self.amp)
                        delta_full = val_nll_cond_full - val_nll_marg
                        print(f"[full-eval@rollback] val(full)={val_nll_cond_full:.6f} Δ(full)={delta_full:.6f}")
                        break
                    if best_overall['state'] is not None:
                        conditional.load_state_dict(best_overall['state'])
                        print("Processing...")
                    if st['name'] == "Stage1":
                        opt, sch = self.build_optim(conditional, lr_adapter=new_lr, lr_backbone=0.0)
                    elif st['name'] == "Stage2":
                        opt, sch = self.build_optim(conditional,
                                                    lr_adapter=new_lr,
                                                    lr_backbone=new_lr*0.2)
                    else:
                        opt, sch = self.build_optim(conditional,
                                                    lr_adapter=new_lr,
                                                    lr_backbone=new_lr*0.1)

                if delta_sub < best_stage['delta']:
                    best_stage.update(dict(
                        nll=val_nll_cond_sub,
                        state={k: v.detach().cpu().clone() for k, v in conditional.state_dict().items()},
                        epoch=total_epoch,
                        delta=delta_sub
                    ))
                if delta_sub < best_overall['delta']:
                    best_overall.update(dict(
                        nll=val_nll_cond_sub,
                        state={k: v.detach().cpu().clone() for k, v in conditional.state_dict().items()},
                        epoch=total_epoch,
                        delta=delta_sub,
                        stage=st['name']
                    ))

                do_full = ((ep + 1) % max(1, self.val_full_every) == 0) or (ep == st['epochs'] - 1)
                if do_full:
                    val_nll_cond_full = eval_nll_cond(conditional, x_val, y_val, batch_size=eval_bs, use_amp=self.amp)
                    delta_full = val_nll_cond_full - val_nll_marg
                    print(f"[full-eval] epoch={total_epoch} val(full)={val_nll_cond_full:.6f} Δ(full)={delta_full:.6f}")
                    if delta_full < best_overall['delta']:
                        best_overall.update(dict(
                            nll=val_nll_cond_full,
                            state={k: v.detach().cpu().clone() for k, v in conditional.state_dict().items()},
                            epoch=total_epoch,
                            delta=delta_full,
                            stage=st['name']
                        ))
                    if (ep + 1) % 20 == 0 and best_overall['state'] is not None:
                        checkpoint = {
                            'state_dict': best_overall['state'],
                            'model_state_dict': best_overall['state'],
                            'epoch': best_overall['epoch'],
                            'val_loss': best_overall['nll'],
                            'delta': best_overall['delta'],
                            'stage': best_overall['stage'],
                            'config': self.cfg,
                            'model_kwargs': self.cfg['model_kwargs'],
                            'target_data_signature': self.cfg['target_data_signature']
                        }
                        torch.save(checkpoint, out_path)
                        print("Processing...")

                if stopper.step(val_nll_cond_sub):
                    print("Processing...")
                    break

                lr_cur_min = min(pg['lr'] for pg in opt.param_groups)
                pbar.set_postfix(train=round(tr_loss, 4),
                                 val_sub=round(val_nll_cond_sub, 4),
                                 delta_sub=round(delta_sub, 4),
                                 lr=f"{lr_cur_min:.6f}")

            val_nll_cond_full = eval_nll_cond(conditional, x_val, y_val, batch_size=eval_bs, use_amp=self.amp)
            delta_full = val_nll_cond_full - val_nll_marg
            print(f"[stage-end] {st['name']}  val(full)={val_nll_cond_full:.6f}  Δ(full)={delta_full:.6f}")
            print("Processing...")

            if stop_training:
                print("Processing...")
                break
            if advance_stage:
                print("Processing...")
                continue

        if best_overall['state'] is not None:
            checkpoint = {
                'state_dict': best_overall['state'],
                'model_state_dict': best_overall['state'],
                'epoch': best_overall['epoch'],
                'val_loss': best_overall['nll'],
                'delta': best_overall['delta'],
                'stage': best_overall['stage'],
                'config': self.cfg,
                'model_kwargs': self.cfg['model_kwargs'],
                'target_data_signature': self.cfg['target_data_signature']
            }
            torch.save(checkpoint, out_path)
        print("Processing...")
        print(f"[best] ValNLL={best_overall['nll']:.6f}  Δ={best_overall['delta']:.6f}  @ {best_overall['stage']} epoch={best_overall['epoch']}")
        return best_overall


def load_xy(data_dir: Path):
    x_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'Y_train.npy')
    x_val   = np.load(data_dir / 'X_val.npy')
    y_val   = np.load(data_dir / 'Y_val.npy')
    return x_train, y_train, x_val, y_val


def main():
    parser = argparse.ArgumentParser(description="FLARE command line utility")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--marginal_model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        vis = torch.cuda.device_count()
        use = args.gpu if args.gpu is not None and 0 <= args.gpu < vis else 0
        device = torch.device(f"cuda:{use}")
        names = [torch.cuda.get_device_name(i) for i in range(vis)]
        print(f"[device] {device} | visible={vis} | names={names}")
    else:
        device = torch.device("cpu")
        print("[device] cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault('seed', 42)
    cfg.setdefault('batch_size', 256)
    cfg.setdefault('grad_accum_steps', 1)
    cfg.setdefault('weight_decay', 0.0)
    cfg.setdefault('use_actnorm', True)
    cfg.setdefault('use_inv1x1', False)
    cfg.setdefault('spline_bins', 8)
    cfg.setdefault('hidden_mult', 4)
    cfg.setdefault('layers', 12)
    cfg.setdefault('eval_bs', 8192)
    cfg.setdefault('cond_hidden', 512)
    cfg.setdefault('dropout', 0.0)
    cfg.setdefault('amp', True)
    cfg.setdefault('adapter_rank', 64)
    cfg.setdefault('adapter_scale', 2e-3)

    cfg.setdefault('val_subset_ratio', 0.1)
    cfg.setdefault('val_subset_min', 1024)
    cfg.setdefault('val_subset_size', 0)
    cfg.setdefault('val_full_every', 10)

    base_lr_cfg = cfg.get('base_lr', 0.012)
    cfg['base_lr'] = base_lr_cfg

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    data_dir = Path(args.data_dir)
    x_tr, y_tr, x_va, y_va = load_xy(data_dir)
    if len(x_tr) != len(y_tr) or len(x_va) != len(y_va) or not len(y_tr) or not len(y_va):
        raise ValueError('X/Y rows must match and training/validation must be nonempty')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] train x={x_tr.shape}, y={y_tr.shape} | val x={x_va.shape}, y={y_va.shape}")

    x_tr_t = torch.from_numpy(x_tr).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).float().to(device)
    x_va_t = torch.from_numpy(x_va).float().to(device)
    y_va_t = torch.from_numpy(y_va).float().to(device)

    marginal, marg_ckpt, backbone = load_flow(
        args.marginal_model, 'marginal', y_dim=y_tr.shape[1], device=device)
    signature = target_data_signature(data_dir)
    if marg_ckpt.get('target_data_signature') != signature:
        raise ValueError('Marginal checkpoint needs matching target train/validation data; retrain it with this release')
    # The conditional flow is a context adapter around exactly this backbone.
    for key in ('layers', 'use_actnorm', 'use_inv1x1'):
        cfg[key] = backbone[key]
    cfg['hidden_mult'] = backbone['hidden'] / backbone['y_dim']
    cfg['spline_bins'] = backbone['bins']
    cfg['target_data_signature'] = signature
    cfg['model_kwargs'] = dict(backbone, x_dim=int(x_tr.shape[1]),
                               dropout=float(cfg.get('dropout', 0.0)), output_scale=0.1,
                               adapter_rank=int(cfg['adapter_rank']),
                               adapter_scale=float(cfg['adapter_scale']))

    conditional = ConditionalFlow(**cfg['model_kwargs']).to(device)

    copy_shared_weights(marginal, conditional)
    zero_init_adapter_B_only(conditional)

    print("Processing...")
    for name, param in conditional.flow.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
            print("  train:", name)
        else:
            param.requires_grad = False
    print("Processing...")

    trainer = ThreeStageTrainer(cfg, device, base_lr=cfg['base_lr'], min_lr=1e-7)
    trainer.fit(marginal, conditional, x_tr_t, y_tr_t, x_va_t, y_va_t, args.out)
    print("Processing...")


if __name__ == '__main__':
    main()
