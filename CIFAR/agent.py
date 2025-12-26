# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import config
from nl_enhancements import SurpriseGateFast, test_time_memorize_fast

# ===================================================================
# GPU-Resident Replay Buffer (Now stores Task IDs)
# ===================================================================
class GPUReplayBuffer:
    def __init__(self, capacity, x_shape, device=config.DEVICE):
        self.capacity = capacity
        self.device = device
        self.x_buffer = torch.zeros((capacity, *x_shape), device=device, dtype=torch.float32)
        self.y_buffer = torch.zeros(capacity, device=device, dtype=torch.long)
        # NEW: Store Task ID for every sample to allow context switching
        self.t_buffer = torch.zeros(capacity, device=device, dtype=torch.long)
        self.ptr = 0
        self.size = 0

    def add(self, x, y, task_id):
        n = x.size(0)
        # Cap batch size to capacity
        if n > self.capacity:
            x = x[-self.capacity:]
            y = y[-self.capacity:]
            n = self.capacity

        indices = torch.arange(self.ptr, self.ptr + n, device=self.device) % self.capacity
        self.x_buffer[indices] = x
        self.y_buffer[indices] = y
        # Fill with the current task_id
        self.t_buffer[indices] = torch.full((n,), task_id, device=self.device, dtype=torch.long)
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            return None, None, None
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.x_buffer[idx], self.y_buffer[idx], self.t_buffer[idx]


# ===================================================================
# Tiny Per-Class Replay Buffer (DER++ style)
# ===================================================================
class TinyPerClassReplayBuffer:
    def __init__(self, samples_per_class, num_classes, x_shape, device=config.DEVICE):
        """
        Maintains a small fixed number of samples per class (e.g., 10-20).
        This is inspired by DER++ and helps prevent catastrophic forgetting.
        """
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.device = device
        self.x_shape = x_shape
        
        # Initialize buffers for each class
        self.class_buffers = []
        for _ in range(num_classes):
            x_buf = torch.zeros((samples_per_class, *x_shape), device=device, dtype=torch.float32)
            y_buf = torch.full((samples_per_class,), -1, device=device, dtype=torch.long)  # -1 means empty
            t_buf = torch.full((samples_per_class,), -1, device=device, dtype=torch.long)
            self.class_buffers.append((x_buf, y_buf, t_buf))
        
        # Track how many samples are stored per class
        self.class_counts = torch.zeros(num_classes, device=device, dtype=torch.long)
        
    def add(self, x, y, task_id):
        """
        Add samples to the buffer, maintaining at most samples_per_class per class.
        Uses reservoir sampling to maintain a random subset.
        """
        for i in range(x.size(0)):
            x_i = x[i]
            y_i = y[i].item()
            task_id_i = task_id
            
            if y_i < 0 or y_i >= self.num_classes:
                continue  # Skip invalid class indices
            
            x_buf, y_buf, t_buf = self.class_buffers[y_i]
            count = self.class_counts[y_i].item()
            
            if count < self.samples_per_class:
                # Buffer not full for this class, add to next empty slot
                x_buf[count] = x_i
                y_buf[count] = y_i
                t_buf[count] = task_id_i
                self.class_counts[y_i] += 1
            else:
                # Buffer full, use reservoir sampling: replace with probability 1/count
                # Since we're adding one at a time, we can think of it as:
                # With probability samples_per_class/(count+1), replace a random existing sample
                if torch.rand(1).item() < self.samples_per_class / (count + 1):
                    replace_idx = torch.randint(0, self.samples_per_class, (1,)).item()
                    x_buf[replace_idx] = x_i
                    y_buf[replace_idx] = y_i
                    t_buf[replace_idx] = task_id_i
    
    def sample(self, batch_size, balanced=True):
        """
        Sample from the buffer. If balanced=True, sample equal number from each class.
        Returns (x, y, task_id) or (None, None, None) if buffer is empty.
        """
        total_samples = self.class_counts.sum().item()
        if total_samples == 0:
            return None, None, None
        
        if balanced:
            # Sample equal number from each class that has samples
            classes_with_samples = [c for c in range(self.num_classes) if self.class_counts[c] > 0]
            if not classes_with_samples:
                return None, None, None
            
            samples_per_class = max(1, batch_size // len(classes_with_samples))
            x_list, y_list, t_list = [], [], []
            
            for class_idx in classes_with_samples:
                x_buf, y_buf, t_buf = self.class_buffers[class_idx]
                count = self.class_counts[class_idx].item()
                
                if count > 0:
                    # Sample from this class's buffer
                    sample_indices = torch.randint(0, count, (min(samples_per_class, count),), device=self.device)
                    x_list.append(x_buf[sample_indices])
                    y_list.append(y_buf[sample_indices])
                    t_list.append(t_buf[sample_indices])
            
            if not x_list:
                return None, None, None
            
            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)
            t = torch.cat(t_list, dim=0)
            
            # If we have more samples than needed, randomly select
            if x.size(0) > batch_size:
                indices = torch.randperm(x.size(0), device=self.device)[:batch_size]
                x = x[indices]
                y = y[indices]
                t = t[indices]
            
            return x, y, t
        else:
            # Sample uniformly from all stored samples
            # Flatten all buffers
            all_x, all_y, all_t = [], [], []
            for class_idx in range(self.num_classes):
                count = self.class_counts[class_idx].item()
                if count > 0:
                    x_buf, y_buf, t_buf = self.class_buffers[class_idx]
                    all_x.append(x_buf[:count])
                    all_y.append(y_buf[:count])
                    all_t.append(t_buf[:count])
            
            if not all_x:
                return None, None, None
            
            x = torch.cat(all_x, dim=0)
            y = torch.cat(all_y, dim=0)
            t = torch.cat(all_t, dim=0)
            
            # Sample from all
            if x.size(0) <= batch_size:
                return x, y, t
            else:
                indices = torch.randint(0, x.size(0), (batch_size,), device=self.device)
                return x[indices], y[indices], t[indices]
    
    def get_class_samples(self, class_idx):
        """Get all samples for a specific class."""
        if class_idx < 0 or class_idx >= self.num_classes:
            return None, None, None
        
        count = self.class_counts[class_idx].item()
        if count == 0:
            return None, None, None
        
        x_buf, y_buf, t_buf = self.class_buffers[class_idx]
        return x_buf[:count], y_buf[:count], t_buf[:count]
    
    def has_samples(self):
        """Check if buffer has any samples."""
        return self.class_counts.sum().item() > 0
    
    def get_total_samples(self):
        """Get total number of samples in buffer."""
        return self.class_counts.sum().item()


# ===================================================================
# CMS (Component) with Nested Learning
# ===================================================================
class CMS(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, freqs):
        super().__init__()
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in freqs
        ])
        self.freqs = freqs
        self.gate = SurpriseGateFast(config.SURPRISE_THRESHOLD)
        self.replay_buffers = [
            GPUReplayBuffer(config.BUFFER_CAPACITY, (input_dim,)) 
            for _ in freqs
        ]
        self.loss_fn = nn.CrossEntropyLoss()

    # UPDATED: Now accepts task_id
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
            # Add to buffer with probability (20%) and include task_id
            if torch.rand(1).item() < 0.2: 
                self.replay_buffers[level_idx].add(x.detach(), y.detach(), task_id)
            return loss.item(), True
        return loss.item(), False

    def consolidate(self, from_idx, to_idx, optimizer, scaler):
        # Consolidation is internal; we ignore task IDs here
        bx, by, _ = self.replay_buffers[from_idx].sample(config.CONSOLIDATION_BATCH_SIZE)
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
# DCC Agent (The Main Agent Class)
# ===================================================================
class DCCAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast = CMS(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_CLASSES, config.FAST_FREQS)
        self.slow = CMS(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_CLASSES, config.SLOW_FREQS)
        
        # Get weight decay from config if available
        weight_decay = getattr(config, 'WEIGHT_DECAY', 0.0)
        
        self.opt_fast = [optim.AdamW(l.parameters(), lr=config.FAST_LR, 
                                    weight_decay=weight_decay, fused=True) 
                        for l in self.fast.levels]
        self.opt_slow = optim.AdamW(self.slow.levels[0].parameters(), lr=config.SLOW_LR,
                                   weight_decay=weight_decay, fused=True)
        self.opt_cons = optim.AdamW(self.fast.levels[0].parameters(), lr=5e-4,
                                   weight_decay=weight_decay, fused=True)
        self.scaler = GradScaler(device='cuda')
        self.steps = 0
        self.gradient_clip = getattr(config, 'GRADIENT_CLIP', None)
        
        # Tiny per-class replay buffer (DER++ style)
        samples_per_class = getattr(config, 'TINY_BUFFER_SAMPLES_PER_CLASS', 15)  # 10-20 as suggested
        self.tiny_buffer = TinyPerClassReplayBuffer(
            samples_per_class=samples_per_class,
            num_classes=config.NUM_CLASSES,
            x_shape=(config.INPUT_DIM,),
            device=config.DEVICE
        )
        
        # Consolidation optimizer for tiny buffer replay
        self.opt_tiny = optim.AdamW(
            list(self.fast.levels[0].parameters()) + list(self.slow.levels[0].parameters()),
            lr=getattr(config, 'CONSOLIDATION_LR', 1e-3),
            weight_decay=weight_decay,
            fused=True
        )

    # UPDATED: Now accepts task_id
    def adapt(self, x, y, task_id, is_new_task=False):
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
            
            with autocast(device_type='cuda'):
                pred = self.slow.levels[0](x)
                loss = self.slow.loss_fn(pred, y)
            self.opt_slow.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Apply gradient clipping if configured
            if self.gradient_clip is not None:
                self.scaler.unscale_(self.opt_slow)
                torch.nn.utils.clip_grad_norm_(self.slow.levels[0].parameters(), self.gradient_clip)
            
            self.scaler.step(self.opt_slow)
            any_stepped = True

        if any_stepped:
            self.scaler.update()
        
        # Add to tiny buffer with higher probability for important samples
        # Add all samples from the batch to ensure we have good coverage
        self.tiny_buffer.add(x.detach(), y.detach(), task_id)

        return loss_sum / len(self.fast.levels)
    
    def consolidate_with_tiny_buffer(self, epochs=1, batch_size=32):
        """
        Perform consolidation using the tiny per-class replay buffer.
        This helps prevent catastrophic forgetting by replaying a balanced
        subset of all seen classes.
        """
        if not self.tiny_buffer.has_samples():
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        # Determine number of batches based on total samples
        total_samples = self.tiny_buffer.get_total_samples()
        if total_samples == 0:
            return 0.0
        
        # Use balanced sampling to ensure all classes are represented
        for _ in range(epochs):
            # Sample balanced batch from tiny buffer
            x_batch, y_batch, t_batch = self.tiny_buffer.sample(
                batch_size=min(batch_size, total_samples),
                balanced=True
            )
            
            if x_batch is None:
                continue
            
            # Train on the replay batch
            with autocast(device_type='cuda'):
                # Forward pass through both fast and slow networks
                fast_out = self.fast(x_batch, 0)
                slow_out = self.slow(x_batch, 0)
                combined_out = config.FAST_WEIGHT * fast_out + config.SLOW_WEIGHT * slow_out
                
                loss = nn.CrossEntropyLoss()(combined_out, y_batch)
            
            # Backward pass
            self.opt_tiny.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Apply gradient clipping if configured
            if self.gradient_clip is not None:
                self.scaler.unscale_(self.opt_tiny)
                torch.nn.utils.clip_grad_norm_(
                    list(self.fast.levels[0].parameters()) + list(self.slow.levels[0].parameters()),
                    self.gradient_clip
                )
            
            self.scaler.step(self.opt_tiny)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)

    def sample_memory(self, batch_size):
        # Returns (x, y, task_id)
        return self.fast.replay_buffers[0].sample(batch_size)

    def get_pooled_feature(self, x):
        """
        Extract pooled features from the DCC model for multimodal integration.
        Returns features of dimension config.HIDDEN_DIM (default: 512).
        """
        # Extract features from the first layer of the fast CMS
        # The fast CMS has levels[0] which is nn.Sequential(Linear, ReLU, Linear)
        # We take the output after the first linear layer (before ReLU)
        with torch.no_grad():
            # Get the first linear layer from the first level
            first_layer = self.fast.levels[0][0]  # nn.Linear
            features = first_layer(x)  # [B, HIDDEN_DIM]
            return features

    def forward(self, x, y=None, memorize_steps=0):
        if memorize_steps > 0 and not self.training:
            test_time_memorize_fast(self.fast, x, y, steps=memorize_steps)
        fast_out = self.fast(x, 0)
        slow_out = self.slow(x, 0)
        return config.FAST_WEIGHT * fast_out + config.SLOW_WEIGHT * slow_out
