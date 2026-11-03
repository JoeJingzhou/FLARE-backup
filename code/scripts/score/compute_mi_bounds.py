"""



    python compute_MI_bounds.py \
        --cond_dir /path/to/data/dataset/bge_to_GritLM_no_pca \
        --gpu 1
"""

import numpy as np
import torch
import json
import argparse
from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flare.flows.marginal import MarginalFlow
from flare.flows.conditional import ConditionalFlow
from flare.checkpoints import load_flow


class MarginalBoundCalculator:
    """Release pipeline helper."""
    
    def __init__(self, model_path, data_dir, confidence=0.95, fast_pca=False, device='cuda'):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.confidence = confidence
        self.delta = 1 - confidence
        self.fast_pca = fast_pca
        self.device = device
        
        self.Y_train = np.load(self.data_dir / 'Y_train.npy')
        self.Y_val = np.load(self.data_dir / 'Y_val.npy')
        
        self.d = self.Y_train.shape[1]
        self.m = len(self.Y_train)
        self.m_val = len(self.Y_val)
        
        self._load_model()
        
        from sklearn.decomposition import PCA
        self._compute_intrinsic_dimensions(PCA, fast_mode=self.fast_pca)
        self._compute_spectral_norms()
        self._compute_actual_gap()
    
    def _load_model(self):
        """Release pipeline helper."""
        self.model, checkpoint, _ = load_flow(self.model_path, 'marginal',
                                               y_dim=self.d, device=self.device)
        self.config = checkpoint['cfg']
        
        self.L = self.config['layers']
        self.bins = self.config['spline_bins']
        
    
    def _compute_intrinsic_dimensions(self, PCA, fast_mode=False):
        """Release pipeline helper."""
        if fast_mode:
            n_components = min(1500, self.d, self.m)
            pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        else:
            pca = PCA()
        
        pca.fit(self.Y_train)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        self.d_90 = np.argmax(cumsum_var >= 0.90) + 1
        self.d_eff = self.d_90
    
    def _compute_spectral_norms(self):
        """Release pipeline helper."""
        spectral_norms = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                if weight.dim() >= 2:
                    if weight.size(0) > 1 and weight.size(1) > 1:
                        S = torch.linalg.svdvals(weight)
                        spectral_norm = S.max().item()
                        spectral_norms.append(spectral_norm)
        
        self.spectral_norms = np.array(spectral_norms)
        self.sigma_mean = np.mean(self.spectral_norms)
        self.L_lipschitz = len(spectral_norms)  # K in the formula
    
    def _compute_actual_gap(self):
        """Release pipeline helper."""
        train_tensor = torch.from_numpy(self.Y_train).float().to(self.device)
        val_tensor = torch.from_numpy(self.Y_val).float().to(self.device)
        
        with torch.no_grad():
            batch_size = 256
            
            train_nlls = []
            for i in range(0, len(train_tensor), batch_size):
                batch = train_tensor[i:i+batch_size]
                nlls = -self.model.log_prob(batch)
                train_nlls.append(nlls.cpu().numpy())
            train_nlls = np.concatenate(train_nlls)
            
            val_nlls = []
            for i in range(0, len(val_tensor), batch_size):
                batch = val_tensor[i:i+batch_size]
                nlls = -self.model.log_prob(batch)
                val_nlls.append(nlls.cpu().numpy())
            val_nlls = np.concatenate(val_nlls)
        
        self.train_nll = np.mean(train_nlls)
        self.val_nll = np.mean(val_nlls)
        self.actual_gap = self.val_nll - self.train_nll
        
        self.train_nlls = train_nlls
        self.val_nlls = val_nlls
    
    def compute_bound(self, C_Rad=12.0):
        """Release pipeline helper."""
        main_term = (self.L_lipschitz * self.sigma_mean * np.sqrt(self.d_eff)) / np.sqrt(self.m)
        rademacher_term = 2 * C_Rad * main_term
        
        M_val = np.percentile(self.val_nlls, 99) - np.percentile(self.val_nlls, 1)
        sampling_error = M_val * np.sqrt(np.log(2/self.delta) / (2 * self.m_val))
        
        M_train = np.percentile(self.train_nlls, 99) - np.percentile(self.train_nlls, 1)
        confidence_term = 3 * M_train * np.sqrt(np.log(2/self.delta) / (2 * self.m))
        
        bound = rademacher_term + sampling_error + confidence_term
        
        return {
            'bound': float(bound),
            'actual_gap': float(self.actual_gap),
            'ratio': float(bound / self.actual_gap) if self.actual_gap > 0 else np.inf,
            'rademacher_term': float(rademacher_term),
            'sampling_error': float(sampling_error),
            'confidence_term': float(confidence_term),
            'M_val': float(M_val),
            'M_train': float(M_train),
            'd_eff': int(self.d_eff),
            'K': int(self.L_lipschitz),
            'sigma_mean': float(self.sigma_mean)
        }


class ConditionalBoundCalculator:
    """Release pipeline helper."""
    
    def __init__(self, model_path, data_dir, confidence=0.95, fast_pca=False, device='cuda'):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.confidence = confidence
        self.delta = 1 - confidence
        self.fast_pca = fast_pca
        self.device = device
        
        data_source_dir = self._find_data_dir()
        self.X_train = np.load(data_source_dir / 'X_train.npy')
        self.Y_train = np.load(data_source_dir / 'Y_train.npy')
        self.X_val = np.load(data_source_dir / 'X_val.npy')
        self.Y_val = np.load(data_source_dir / 'Y_val.npy')
        
        self.x_dim = self.X_train.shape[1]
        self.y_dim = self.Y_train.shape[1]
        self.m = len(self.Y_train)
        self.m_val = len(self.Y_val)
        
        self._load_model()
        
        from sklearn.decomposition import PCA
        self._compute_intrinsic_dimensions(PCA, fast_mode=self.fast_pca)
        self._compute_spectral_norms()
        self._compute_actual_gap()
    
    def _find_data_dir(self):
        """
        
        """
        if (self.data_dir / 'X_train.npy').exists():
            return self.data_dir
        
        dir_name = self.data_dir.name
        parent = self.data_dir.parent
        
        if dir_name.endswith('_no_pca') and not dir_name.endswith('_sent1only_no_pca'):
            alt_name = dir_name[:-7] + '_sent1only_no_pca'
            alt_dir = parent / alt_name
            if (alt_dir / 'X_train.npy').exists():
                print("Processing...")
                return alt_dir
        elif dir_name.endswith('_sent1only_no_pca'):
            alt_name = dir_name[:-17] + '_no_pca'
            alt_dir = parent / alt_name
            if (alt_dir / 'X_train.npy').exists():
                print("Processing...")
                return alt_dir
        
        return self.data_dir
    
    def _load_model(self):
        """Release pipeline helper."""
        self.model, checkpoint, _ = load_flow(self.model_path, 'conditional',
                                               y_dim=self.y_dim, x_dim=self.x_dim, device=self.device)
        self.config = checkpoint['config']
        
        self.L = self.config['layers']
        self.bins = self.config['spline_bins']
        
    
    def _compute_intrinsic_dimensions(self, PCA, fast_mode=False):
        """Release pipeline helper."""
        if fast_mode:
            n_components = min(1500, self.y_dim, self.m)
            pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        else:
            pca = PCA()
        
        pca.fit(self.Y_train)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        self.d_90 = np.argmax(cumsum_var >= 0.90) + 1
        self.d_eff = self.d_90
    
    def _compute_spectral_norms(self):
        """Release pipeline helper."""
        spectral_norms = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                if weight.dim() >= 2:
                    if weight.size(0) > 1 and weight.size(1) > 1:
                        S = torch.linalg.svdvals(weight)
                        spectral_norm = S.max().item()
                        spectral_norms.append(spectral_norm)
        
        self.spectral_norms = np.array(spectral_norms)
        self.sigma_mean = np.mean(self.spectral_norms)
        self.L_lipschitz = len(spectral_norms)
    
    def _compute_actual_gap(self):
        """Release pipeline helper."""
        x_train_t = torch.from_numpy(self.X_train).float().to(self.device)
        y_train_t = torch.from_numpy(self.Y_train).float().to(self.device)
        x_val_t = torch.from_numpy(self.X_val).float().to(self.device)
        y_val_t = torch.from_numpy(self.Y_val).float().to(self.device)
        
        with torch.no_grad():
            batch_size = 256
            
            train_nlls = []
            for i in range(0, len(y_train_t), batch_size):
                x_batch = x_train_t[i:i+batch_size]
                y_batch = y_train_t[i:i+batch_size]
                nlls = -self.model.log_prob(y_batch, x_batch)
                train_nlls.append(nlls.cpu().numpy())
            train_nlls = np.concatenate(train_nlls)
            
            val_nlls = []
            for i in range(0, len(y_val_t), batch_size):
                x_batch = x_val_t[i:i+batch_size]
                y_batch = y_val_t[i:i+batch_size]
                nlls = -self.model.log_prob(y_batch, x_batch)
                val_nlls.append(nlls.cpu().numpy())
            val_nlls = np.concatenate(val_nlls)
        
        self.train_nll = np.mean(train_nlls)
        self.val_nll = np.mean(val_nlls)
        self.actual_gap = self.val_nll - self.train_nll
        
        self.train_nlls = train_nlls
        self.val_nlls = val_nlls
    
    def compute_bound(self, C_Rad=12.0):
        """Release pipeline helper."""
        main_term = (self.L_lipschitz * self.sigma_mean * np.sqrt(self.d_eff)) / np.sqrt(self.m)
        rademacher_term = 2 * C_Rad * main_term
        
        M_val = np.percentile(self.val_nlls, 99) - np.percentile(self.val_nlls, 1)
        sampling_error = M_val * np.sqrt(np.log(2/self.delta) / (2 * self.m_val))
        
        M_train = np.percentile(self.train_nlls, 99) - np.percentile(self.train_nlls, 1)
        confidence_term = 3 * M_train * np.sqrt(np.log(2/self.delta) / (2 * self.m))
        
        bound = rademacher_term + sampling_error + confidence_term
        
        return {
            'bound': float(bound),
            'actual_gap': float(self.actual_gap),
            'ratio': float(bound / self.actual_gap) if self.actual_gap > 0 else np.inf,
            'rademacher_term': float(rademacher_term),
            'sampling_error': float(sampling_error),
            'confidence_term': float(confidence_term),
            'M_val': float(M_val),
            'M_train': float(M_train),
            'd_eff': int(self.d_eff),
            'K': int(self.L_lipschitz),
            'sigma_mean': float(self.sigma_mean)
        }


def parse_model_names(cond_dir_name):
    """
    
          -> source: bge_multilingual_gemma2
          -> target: GritLM_7B
    
    """
    name = cond_dir_name
    for suffix in ['_sent1only_no_pca', '_no_pca', '_marginal']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    if '_to_' not in name:
        raise ValueError("Invalid input format")
    
    parts = name.split('_to_')
    source = parts[0]
    target = parts[1]
    
    return source, target


def compute_MI_generalization_bound(cond_dir, C_Rad=12.0, confidence=0.95, 
                                   fast_pca=False, device='cuda', verbose=True):
    """
    
    Args:
    
    Returns:
    """
    cond_dir = Path(cond_dir)
    
    if verbose:
        print("=" * 70)
        print("Processing...")
        print("=" * 70)
        print("Processing...")
    
    source_name, target_name = parse_model_names(cond_dir.name)
    
    if verbose:
        print(f"Source (X): {source_name}")
        print(f"Target (U): {target_name}")
        print("Processing...")
    
    dataset_dir = cond_dir.parent
    
    possible_suffixes = ['_sent1only_no_pca', '_no_pca', '_marginal']
    marg_dir = None
    
    for suffix in possible_suffixes:
        test_dir = dataset_dir / f"{target_name}{suffix}"
        if test_dir.exists():
            marg_dir = test_dir
            break
    
    if marg_dir is None:
        tried = [f"{dataset_dir}/{target_name}{s}" for s in possible_suffixes]
        raise FileNotFoundError("Marginal model directory not found. Tried: " + ", ".join(tried))
    
    cond_model_path = cond_dir / "cond_flow.pt"
    marg_model_path = marg_dir / "marginal_model.pth"
    
    if not cond_model_path.exists():
        raise FileNotFoundError("Required file or directory not found")
    if not marg_model_path.exists():
        raise FileNotFoundError("Required file or directory not found")
    
    if verbose:
        print("=" * 70)
        print("Processing...")
        print("=" * 70)
    
    cond_calc = ConditionalBoundCalculator(
        cond_model_path, cond_dir, confidence, fast_pca, device
    )
    cond_result = cond_calc.compute_bound(C_Rad)
    
    if verbose:
        print("Processing...")
        print("Processing...")
        print("Processing...")
        
        print("=" * 70)
        print("Processing...")
        print("=" * 70)
    
    marg_calc = MarginalBoundCalculator(
        marg_model_path, marg_dir, confidence, fast_pca, device
    )
    marg_result = marg_calc.compute_bound(C_Rad)
    
    if verbose:
        print("Processing...")
        print("Processing...")
        print("Processing...")
        
        print("=" * 70)
        print("Processing...")
        print("=" * 70)
    
    MI_bound = cond_result['bound'] + marg_result['bound']
    
    actual_MI_gap = cond_result['actual_gap'] + marg_result['actual_gap']
    
    MI_ratio = MI_bound / actual_MI_gap if actual_MI_gap > 0 else np.inf
    
    if verbose:
        print("Processing...")
        print(f"  Δ_MI ≤ Δ_cond + Δ_marg")
        print(f"       ≤ {cond_result['bound']:.4f} + {marg_result['bound']:.4f}")
        print(f"       ≤ {MI_bound:.4f}")
        print("Processing...")
        print(f"  gap_cond + gap_marg")
        print(f"  = {cond_result['actual_gap']:.4f} + {marg_result['actual_gap']:.4f}")
        print(f"  = {actual_MI_gap:.4f}")
        print("Processing...")
        print("Processing...")
        print("Processing...")
        print("Processing...")
    
    result = {
        'MI_bound': float(MI_bound),
        'actual_MI_gap': float(actual_MI_gap),
        'ratio': float(MI_ratio),
        'source_model': source_name,
        'target_model': target_name,
        'conditional': cond_result,
        'marginal': marg_result,
        'parameters': {
            'C_Rad': float(C_Rad),
            'confidence': float(confidence),
            'delta': float(1 - confidence)
        }
    }
    
    output_path_local = cond_dir / "MI_generalization_bounds.json"
    with open(output_path_local, 'w') as f:
        json.dump(result, f, indent=2)
    
    # cond_dir: .../data/dataset_name/model_to_model_no_pca
    project_root = cond_dir.parent.parent.parent
    dataset_name = cond_dir.parent.name
    analysis_dir = project_root / "analysis_results" / dataset_name / "MI_bounds"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{source_name}_to_{target_name}.json"
    output_path_analysis = analysis_dir / output_filename
    with open(output_path_analysis, 'w') as f:
        json.dump(result, f, indent=2)
    
    if verbose:
        print("Processing...")
        print(f"   - {output_path_local}")
        print(f"   - {output_path_analysis}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="FLARE command line utility")
    parser.add_argument('--cond_dir', type=str, required=True,
                       help="See README for details")
    parser.add_argument('--C_Rad', type=float, default=12.0,
                       help="See README for details")
    parser.add_argument('--confidence', type=float, default=0.95,
                       help="See README for details")
    parser.add_argument('--fast_pca', action='store_true',
                       help="See README for details")
    parser.add_argument('--gpu', type=int, default=None,
                       help="See README for details")
    
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    result = compute_MI_generalization_bound(
        cond_dir=args.cond_dir,
        C_Rad=args.C_Rad,
        confidence=args.confidence,
        fast_pca=args.fast_pca,
        device=device,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Processing...")
    print("=" * 70)
    print(f"I({result['target_model']}; {result['source_model']})")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("=" * 70)


if __name__ == '__main__':
    main()

