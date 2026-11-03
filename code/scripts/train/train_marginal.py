import os, argparse, yaml, numpy as np, torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from flare.flows.marginal import MarginalFlow
from flare.utils import EMA, count_params
from flare.checkpoints import marginal_kwargs, target_data_signature


class NLLImprovementStopper:
    """Release pipeline helper."""
    def __init__(self, patience=5, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.previous_nll = None
        self.improvement_count = 0
        
    def step(self, current_nll):
        if self.previous_nll is None:
            self.previous_nll = current_nll
            return False
            
        improvement = self.previous_nll - current_nll
        
        if improvement < self.threshold:
            self.improvement_count += 1
        else:
            self.improvement_count = 0
            
        self.previous_nll = current_nll
        
        return self.improvement_count >= self.patience


class WindowImprovementStopper:
    """Release pipeline helper."""
    def __init__(self, threshold=2.0):
        self.threshold = threshold
        self.previous_window_best = None
        
    def step(self, current_window_best):
        if self.previous_window_best is None:
            self.previous_window_best = current_window_best
            return False
            
        improvement = self.previous_window_best - current_window_best
        
        self.previous_window_best = current_window_best
        
        return improvement < self.threshold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--data_dir', type=str, default='data')
    ap.add_argument('--out', type=str, default='data/marg_flow.pt')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg.get('seed', 42))

    Y_train = np.load(os.path.join(args.data_dir, 'Y_train.npy'))
    Y_val = np.load(os.path.join(args.data_dir, 'Y_val.npy'))
    y_dim = Y_train.shape[1]
    if not len(Y_train) or not len(Y_val):
        raise ValueError('Training and validation splits must both be nonempty')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model_kwargs = marginal_kwargs(cfg, y_dim)
    signature = target_data_signature(args.data_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    model = MarginalFlow(**model_kwargs).to(device)
    print(f"Params: {count_params(model)/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    
    scheduler_type = cfg.get('lr_scheduler', None)
    if scheduler_type == 'StepLR':
        step_size = int(cfg.get('lr_step_size', 10))
        gamma = float(cfg.get('lr_gamma', 0.8))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
        print(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
    elif scheduler_type == 'ReduceLROnPlateau':
        patience = int(cfg.get('lr_patience', 5))
        factor = float(cfg.get('lr_factor', 0.5))
        min_lr = float(cfg.get('lr_min_lr', 1e-6))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=factor, patience=patience, 
            min_lr=min_lr, verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler: patience={patience}, factor={factor}, min_lr={min_lr}")
    else:
        scheduler = None
        print("No learning rate scheduler")
    
    # Initialize ActNorm on training data before taking the first EMA snapshot.
    with torch.no_grad():
        model.log_prob(torch.from_numpy(Y_train[:max(2, cfg['batch_size'])]).float().to(device))
    ema = EMA(model, decay=float(cfg['ema_decay']))
    
    grad_accum = int(cfg.get('grad_accum_steps', 1))

    train_loader = DataLoader(TensorDataset(torch.from_numpy(Y_train).float()), batch_size=cfg['batch_size'], shuffle=True, drop_last=False)
    val_tensor = torch.from_numpy(Y_val).float().to(device)

    best = float('inf')
    best_in_window = float('inf')
    window_best_state = None
    
    nll_improvement_stopper = NLLImprovementStopper(
        patience=cfg.get('nll_improvement_patience', 5),
        threshold=cfg.get('nll_improvement_threshold', 0.001)
    )
    
    window_improvement_stopper = WindowImprovementStopper(
        threshold=cfg.get('window_improvement_threshold', 2.0)
    )
    for epoch in range(cfg['max_epochs']):
        model.train()
        opt.zero_grad()
        step = 0
        for (y,) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            y = y.to(device)
            loss = -model.log_prob(y).mean() / grad_accum
            loss.backward()
            step += 1
            
            if (step % grad_accum == 0) or (step == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                ema.update(model)
        model.eval(); ema.load_shadow(model)
        with torch.no_grad():
            val_ll = model.log_prob(val_tensor).mean().item()
            val_nll = -val_ll
        print(f"Val NLL: {val_nll:.4f}")
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_nll)
                current_lr = opt.param_groups[0]['lr']
                print(f"Current LR: {current_lr:.6f}")
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"Current LR: {current_lr:.6f}")
        
        if val_nll < best:
            best = val_nll
            torch.save({'state_dict': model.state_dict(), 'cfg': cfg,
                        'model_kwargs': model_kwargs, 'target_data_signature': signature}, args.out)
        
        if val_nll < best_in_window:
            best_in_window = val_nll
            window_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            if window_best_state is not None:
                current_state = model.state_dict()
                model.load_state_dict(window_best_state)
                torch.save({'state_dict': model.state_dict(), 'cfg': cfg,
                            'model_kwargs': model_kwargs, 'target_data_signature': signature}, args.out)
                model.load_state_dict(current_state)
            print("Processing...")
            
            if window_improvement_stopper.step(best_in_window):
                print("Processing...")
                break
            
            best_in_window = float('inf')
            window_best_state = None
        
        if nll_improvement_stopper.step(val_nll):
            print("Processing...")
            break
    print(f"Best Val NLL: {best:.4f}; saved to {args.out}")

if __name__ == '__main__':
    main()
