import torch, math
from copy import deepcopy

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
        self.parameter_names = {name for name, _ in model.named_parameters()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.parameter_names:
                self.shadow[k].lerp_(v.detach(), 1 - self.decay)
            else:
                # Permutations and ActNorm initialization flags are state, not averages.
                self.shadow[k].copy_(v.detach())
    @torch.no_grad()
    def load_shadow(self, model):
        model.load_state_dict(self.shadow, strict=True)

def std_normal_logprob(z):
    return -0.5 * (z.pow(2) + math.log(2*math.pi))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
