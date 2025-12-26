
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import math

# ===================================================================
# NL Enhancements
# ===================================================================
class SurpriseGateFast:
    """
    Optimized Surprise Gate using torch._foreach operations for 
    maximum throughput on CUDA.
    """
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    @torch.no_grad()
    def should_update_fast(self, module: nn.Module) -> bool:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads: return False
        global_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
        avg_norm = global_norm / (len(grads) ** 0.5)
        return avg_norm.item() > self.threshold

def test_time_memorize_fast(model, x, y=None, steps=2, inner_opt=None, threshold=0.01):
    model.eval() 
    if inner_opt is None:
        inner_opt = torch.optim.SGD(model.parameters(), lr=0.01)

    with torch.enable_grad():
        for _ in range(steps):
            pred = model(x)
            if y is not None:
                loss = F.cross_entropy(pred, y)
            else:
                probs = F.softmax(pred, dim=1)
                loss = -(probs * (probs + 1e-10).log()).sum(dim=1).mean()

            inner_opt.zero_grad(set_to_none=True)
            loss.backward()

            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads:
                grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
                if grad_norm.item() > threshold:
                    inner_opt.step()
    
    with torch.no_grad():
        out = model(x)
    return out

# ===================================================================
# GPU Replay Buffers
# ===================================================================
class GPUReplayBuffer:
    def __init__(self, capacity, x_shape, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.x_buffer = torch.zeros((capacity, *x_shape), device=device, dtype=torch.float32)
        self.y_buffer = torch.zeros(capacity, device=device, dtype=torch.long)
        self.t_buffer = torch.zeros(capacity, device=device, dtype=torch.long)
        self.ptr = 0
        self.size = 0

    def add(self, x, y, task_id):
        n = x.size(0)
        if n > self.capacity:
            x = x[-self.capacity:]
            y = y[-self.capacity:]
            n = self.capacity

        indices = torch.arange(self.ptr, self.ptr + n, device=self.device) % self.capacity
        self.x_buffer[indices] = x
        self.y_buffer[indices] = y
        self.t_buffer[indices] = torch.full((n,), task_id, device=self.device, dtype=torch.long)
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        if self.size == 0: return None, None, None
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.x_buffer[idx], self.y_buffer[idx], self.t_buffer[idx]

# ===================================================================
# CMS (Component) with Nested Learning
# ===================================================================
class CMS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, freqs, surprise_threshold=0.003, buffer_capacity=2000, device='cuda'):
        super().__init__()
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in freqs
        ])
        self.freqs = freqs
        self.gate = SurpriseGateFast(surprise_threshold)
        self.replay_buffers = [
            GPUReplayBuffer(buffer_capacity, (input_dim,), device=device) 
            for _ in freqs
        ]
        self.loss_fn = nn.CrossEntropyLoss()

    def update(self, x, y, task_id, level_idx, optimizer, scaler):
        with autocast(device_type='cuda'):
            pred = self.levels[level_idx](x)
            loss = self.loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        surprise = self.gate.should_update_fast(self.levels[level_idx])
        random_check = (torch.rand(1).item() < self.freqs[level_idx])
        should_step = surprise or random_check
        
        if should_step:
            scaler.step(optimizer)
            if torch.rand(1).item() < 0.2: 
                self.replay_buffers[level_idx].add(x.detach(), y.detach(), task_id)
            return loss.item(), True
        return loss.item(), False

    def consolidate(self, from_idx, to_idx, optimizer, scaler, batch_size=96):
        bx, by, _ = self.replay_buffers[from_idx].sample(batch_size)
        if bx is None: return False
        
        with autocast(device_type='cuda'):
            pred = self.levels[to_idx](bx)
            loss = self.loss_fn(pred, by)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        return True

    def forward(self, x, level_idx=0):
        return self.levels[level_idx](x)

# ===================================================================
# DCC Agent (Adapted for Lab)
# ===================================================================
class DCCAgent(nn.Module):
    def __init__(self, hp: dict):
        super().__init__()
        
        # Hyperparameters extraction
        self.input_dim = hp.get("input_dim", 3072)
        self.hidden_dim = hp.get("hidden_dim", 512)
        self.num_classes = hp.get("num_classes", 100) # CIFAR 100 default
        
        self.fast_freqs = hp.get("fast_freqs", [0.98, 0.95])
        self.slow_freqs = hp.get("slow_freqs", [0.80])
        
        self.device = hp.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.fast = CMS(self.input_dim, self.hidden_dim, self.num_classes, self.fast_freqs, device=self.device)
        self.slow = CMS(self.input_dim, self.hidden_dim, self.num_classes, self.slow_freqs, device=self.device)
        
        weight_decay = hp.get("weight_decay", 0.0)
        
        self.opt_fast = [optim.AdamW(l.parameters(), lr=hp.get("fast_lr", 0.001), 
                                    weight_decay=weight_decay, fused=False) # Fused needs CUDA often, safety
                        for l in self.fast.levels]
        self.opt_slow = optim.AdamW(self.slow.levels[0].parameters(), lr=hp.get("slow_lr", 0.001),
                                   weight_decay=weight_decay, fused=False)
        self.opt_cons = optim.AdamW(self.fast.levels[0].parameters(), lr=5e-4,
                                   weight_decay=weight_decay, fused=False)
                                   
        self.scaler = GradScaler('cuda' if self.device == 'cuda' else 'cpu', enabled=(self.device=='cuda'))
        self.steps = 0
        
        # We simplify and remove TinyBuffer / Dreaming for this basic port 
        # unless specifically needed, to reduce dependencies. 
        # But user asked for "that code", so let's stick to the core mechanics.
        
        self.fast_weight = hp.get("fast_weight", 0.7)
        self.slow_weight = hp.get("slow_weight", 0.3)

    def adapt(self, x, y, task_id, is_new_task=False):
        # Ensure device
        x = x.to(self.device)
        y = y.to(self.device)
        
        loss_sum = 0
        any_stepped = False
        
        for i, opt in enumerate(self.opt_fast):
            l, stepped = self.fast.update(x, y, task_id, i, opt, self.scaler)
            loss_sum += l
            if stepped: any_stepped = True
        self.steps += 1
        
        if is_new_task or (self.steps % 10 == 0):
            for i in range(1, len(self.fast.levels)):
                s = self.fast.consolidate(i, 0, self.opt_cons, self.scaler)
                if s: any_stepped = True
            
            with autocast(device_type=self.device, enabled=(self.device=='cuda')):
                pred = self.slow.levels[0](x)
                loss = self.slow.loss_fn(pred, y)
            self.opt_slow.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt_slow)
            any_stepped = True

        if any_stepped:
            self.scaler.update()

        return loss_sum / len(self.fast.levels)

    def forward(self, x, y=None, memorize_steps=0):
        x = x.to(self.device)
        if memorize_steps > 0 and not self.training:
            test_time_memorize_fast(self.fast, x, y, steps=memorize_steps)
        
        fast_out = self.fast(x, 0)
        slow_out = self.slow(x, 0)
        return self.fast_weight * fast_out + self.slow_weight * slow_out
