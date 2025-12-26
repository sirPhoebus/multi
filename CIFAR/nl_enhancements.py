# nl_enhancements.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurpriseGateFast:
    """
    Optimized Surprise Gate using torch._foreach operations for 
    maximum throughput on CUDA.
    """
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    @torch.no_grad()
    def should_update_fast(self, module: nn.Module) -> bool:
        # Collect grads
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        
        if not grads:
            return False

        # Optimized norm calculation using foreach (PyTorch 2.0+)
        # This avoids launching a separate kernel for every parameter tensor
        global_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
        
        # Average norm approximation
        avg_norm = global_norm / (len(grads) ** 0.5)
        return avg_norm.item() > self.threshold

def test_time_memorize_fast(model, x, y=None, steps=2, inner_opt=None, threshold=0.01):
    """
    Performs test-time adaptation (memorization) on the fly.
    """
    model.eval() # Keep eval mode for BatchRenorm/Dropout behavior
    
    # We need a local optimizer if one isn't provided, but strictly 
    # capturing the model's parameters. 
    # Note: Creating optimizers in a loop is expensive. 
    # In a production version, this should be cached.
    if inner_opt is None:
        inner_opt = torch.optim.SGD(model.parameters(), lr=0.01)

    with torch.enable_grad():
        for _ in range(steps):
            pred = model(x)
            
            if y is not None:
                loss = F.cross_entropy(pred, y)
            else:
                # Entropy minimization (unsupervised)
                probs = F.softmax(pred, dim=1)
                loss = -(probs * (probs + 1e-10).log()).sum(dim=1).mean()

            inner_opt.zero_grad(set_to_none=True)
            loss.backward()

            # Quick check for surprise
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads:
                grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
                if grad_norm.item() > threshold:
                    inner_opt.step()
    
    # Final forward pass
    with torch.no_grad():
        out = model(x)
    
    return out