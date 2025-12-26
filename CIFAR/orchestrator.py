import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from agent import DCCAgent
import numpy as np

GNN_HIDDEN_DIM = 32
ROUTING_LR = 1e-3
USAGE_PENALTY_WEIGHT = config.USAGE_PENALTY_WEIGHT
BIRTH_THRESHOLD = getattr(config, 'BIRTH_THRESHOLD', 2.0)
USE_VALIDATION_TRIGGER = getattr(config, 'USE_VALIDATION_TRIGGER', False)
VALIDATION_SPAWN_THRESHOLD = getattr(config, 'VALIDATION_SPAWN_THRESHOLD', 0.3)
VALIDATION_CHECK_FREQ = getattr(config, 'VALIDATION_CHECK_FREQ', 5)
CONSOLIDATION_AFTER_TASK = getattr(config, 'CONSOLIDATION_AFTER_TASK', False)
CONSOLIDATION_EPOCHS = getattr(config, 'CONSOLIDATION_EPOCHS', 2)
CONSOLIDATION_BATCHES_PER_EPOCH = getattr(config, 'CONSOLIDATION_BATCHES_PER_EPOCH', 20)

class AgentGraph:
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.adj = torch.eye(num_agents, device=config.DEVICE)

    def expand(self):
        self.num_agents += 1
        new_adj = torch.eye(self.num_agents, device=config.DEVICE)
        new_adj[:self.num_agents-1, :self.num_agents-1] = self.adj
        self.adj = new_adj
    
    def update_interaction(self, agent_idx):
        pass

class RoutingGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj):
        m = torch.matmul(adj, x)
        h = self.fc1(m)
        h = self.norm(h)
        h = F.relu(h)
        scores = self.fc_out(h).squeeze(-1)
        return scores

class GraphOrchestrator(nn.Module):
    def __init__(self, agents):
        super().__init__()
        self.agents = nn.ModuleList(agents)
        self.num_agents = len(agents)
        self.current_task = 0
        
        # Maps Task ID -> Agent ID. 
        # e.g., {0: 0, 1: 1, 2: 2}
        # This prevents the router from being stupid during training.
        self.task_agent_map = {0: 0} 

        self.register_buffer("usage", torch.zeros(self.num_agents))
        self.register_buffer("loss_ma", torch.ones(self.num_agents)) 
        
        self.agent_graph = AgentGraph(self.num_agents)
        input_dim = 3 + config.NUM_TASKS 
        self.routing_gnn = RoutingGNN(input_dim, GNN_HIDDEN_DIM)
        self.routing_opt = torch.optim.AdamW(self.routing_gnn.parameters(), lr=ROUTING_LR)
        self.register_buffer("task_eye", torch.eye(config.NUM_TASKS))

    def set_task(self, task_id):
        self.current_task = int(task_id)

    def _expand_system(self):
        max_agents = getattr(config, 'MAX_AGENTS', 20)
        if self.num_agents >= max_agents:
            print(f"  [NEUROGENESIS LIMIT REACHED] Cannot spawn more than {max_agents} agents. Reusing existing agents.")
            return False

        print(f"  [NEUROGENESIS] Spawning Agent {self.num_agents + 1} for Task {self.current_task}...")
        
        new_agent = DCCAgent().to(config.DEVICE)
        self.agents.append(new_agent)
        self.num_agents += 1
        self.agent_graph.expand()

        new_usage = torch.zeros(1, device=config.DEVICE)
        self.usage = torch.cat([self.usage, new_usage])
        
        new_loss = torch.tensor([0.5], device=config.DEVICE)
        self.loss_ma = torch.cat([self.loss_ma, new_loss])
        
        # CRITICAL FIX: Bind this new agent to the current task immediately
        self.task_agent_map[self.current_task] = self.num_agents - 1
        return True

    def _test_agent_zero_shot(self, agent_idx, validation_loader, task_id):
        """
        Test an existing agent on a new task's validation set with zero-shot (no adaptation).
        Returns the accuracy.
        """
        original_task = self.current_task
        self.current_task = task_id
        
        agent = self.agents[agent_idx]
        agent.eval()  # Set to eval mode for inference
        
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in validation_loader:
                x, y = x.to(config.DEVICE, non_blocking=True), y.to(config.DEVICE, non_blocking=True)
                
                # Forward pass only (zero-shot)
                out = agent(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"    Agent {agent_idx} on Task {task_id}, Zero-shot accuracy = {accuracy:.4f}")
        
        # Restore original task
        self.current_task = original_task
        agent.train()  # Restore train mode
        
        return accuracy

    def _test_agent_on_validation(self, agent_idx, validation_loader, task_id, epochs=2):
        """
        Test an existing agent on a new task's validation set for 1-2 epochs.
        Returns the accuracy after testing.
        """
        original_task = self.current_task
        self.current_task = task_id
        
        agent = self.agents[agent_idx]
        agent.train()  # Keep in train mode to allow adaptation
        
        # Store original optimizer states to restore later if needed
        original_fast_lrs = [pg['lr'] for opt in agent.opt_fast for pg in opt.param_groups]
        original_slow_lrs = [pg['lr'] for pg in agent.opt_slow.param_groups]
        
        accuracies = []
        for epoch in range(epochs):
            correct, total = 0, 0
            for x, y in validation_loader:
                x, y = x.to(config.DEVICE, non_blocking=True), y.to(config.DEVICE, non_blocking=True)
                
                # Adapt the agent on this batch
                agent.adapt(x, y, task_id, is_new_task=(epoch==0))
                
                # Evaluate immediately after adaptation
                with torch.no_grad():
                    out = agent(x)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
            
            accuracy = correct / total if total > 0 else 0.0
            accuracies.append(accuracy)
            print(f"    Agent {agent_idx} on Task {task_id}, Epoch {epoch+1}: accuracy = {accuracy:.4f}")
        
        # Restore original task
        self.current_task = original_task
        
        # Return the best accuracy achieved
        return max(accuracies) if accuracies else 0.0

    def check_reuse_for_new_task(self, validation_loader, task_id):
        """
        Check if any existing agent can handle the new task.
        Tests each agent on the validation set with zero-shot (no adaptation).
        Returns the agent index to reuse, or None if none are suitable.
        """
        print(f"  [REUSE CHECK] Testing {self.num_agents} existing agents on Task {task_id} (zero-shot)...")
        
        zero_shot_threshold = getattr(config, 'REUSE_ZERO_SHOT_THRESHOLD', 0.45)
        adapt_threshold = getattr(config, 'REUSE_ADAPT_THRESHOLD', 0.35)
        reuse_test_epochs = getattr(config, 'REUSE_TEST_EPOCHS', 2)
        
        # First, check zero-shot performance
        best_agent = None
        best_accuracy = 0.0
        
        for agent_idx in range(self.num_agents):
            accuracy = self._test_agent_zero_shot(agent_idx, validation_loader, task_id)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_agent = agent_idx
        
        print(f"  [REUSE CHECK] Best zero-shot agent: {best_agent} with accuracy: {best_accuracy:.4f}")
        
        # If zero-shot accuracy meets threshold, reuse immediately
        if best_accuracy >= zero_shot_threshold:
            print(f"  [REUSE] Agent {best_agent} achieves {best_accuracy:.4f} >= {zero_shot_threshold} (zero-shot), reusing for Task {task_id}")
            # Map this task to the best agent
            self.task_agent_map[task_id] = best_agent
            return best_agent
        
        # Zero-shot failed, check if we should try adaptation
        # According to user feedback: "else: spawn new agent" - so we should NOT do adaptation
        # But keeping adaptation as an option if configured
        if getattr(config, 'REUSE_TRY_ADAPTATION', False):
            print(f"  [REUSE] Zero-shot failed, trying adaptation for {reuse_test_epochs} epochs...")
            best_agent = None
            best_accuracy = 0.0
            
            for agent_idx in range(self.num_agents):
                accuracy = self._test_agent_on_validation(agent_idx, validation_loader, task_id, epochs=reuse_test_epochs)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_agent = agent_idx
            
            print(f"  [REUSE CHECK] Best agent after adaptation: {best_agent} with accuracy: {best_accuracy:.4f}")
            
            if best_accuracy >= adapt_threshold:
                print(f"  [REUSE] Agent {best_agent} achieves {best_accuracy:.4f} >= {adapt_threshold} (after adaptation), reusing for Task {task_id}")
                self.task_agent_map[task_id] = best_agent
                return best_agent
        
        print(f"  [REUSE] No suitable agent found (best zero-shot accuracy: {best_accuracy:.4f} < {zero_shot_threshold})")
        return None

    def _get_node_features(self):
        u_norm = self.usage / (self.usage.max() + 1e-6)
        l_norm = self.loss_ma / (self.loss_ma.max() + 1e-6)
        noise = torch.randn(self.num_agents, 1, device=config.DEVICE) * 0.1
        task_feat = self.task_eye[self.current_task].repeat(self.num_agents, 1)
        return torch.cat([u_norm.unsqueeze(1), l_norm.unsqueeze(1), noise, task_feat], dim=1)

    def _route(self, train=True):
        x = self._get_node_features()
        pred_rewards = self.routing_gnn(x, self.agent_graph.adj)
        
        if train:
            penalty = self.usage * USAGE_PENALTY_WEIGHT
            adjusted_scores = pred_rewards - penalty

            if torch.rand(1).item() < config.ROUTING_EPSILON:
                selected_idx = torch.randint(0, self.num_agents, (1,)).item()
            else:
                selected_idx = adjusted_scores.argmax().item()
            
            target_reward = -self.loss_ma.detach()
            loss_gnn = F.mse_loss(pred_rewards, target_reward)
            self.routing_opt.zero_grad(set_to_none=True)
            loss_gnn.backward()
            self.routing_opt.step()
            return selected_idx
        else:
            return pred_rewards.argmax().item()

    def adapt(self, x, y, is_new_task=False, epoch=0, validation_accuracy=None):
        # 1. Check if we already have an expert for this task
        if self.current_task in self.task_agent_map:
            # FORCE use of the assigned expert. Ignore the router.
            idx = self.task_agent_map[self.current_task]
            
            # (Optional) Still run the router just to train it in the background
            _ = self._route(train=True) 

        else:
            # 2. No expert yet? Use router to see if anyone can handle it
            idx = self._route(train=True)

            # 3. Check for Neurogenesis with improved criterion
            if self.current_task > 0:
                with torch.no_grad():
                    pred = self.agents[idx](x)
                    loss_val = nn.CrossEntropyLoss()(pred, y).item()
                
                should_spawn = False
                
                # Criterion 1: Loss-based threshold (reduced from 4.0 to 2.0)
                if loss_val > BIRTH_THRESHOLD:
                    should_spawn = True
                    print(f"  [NEUROGENESIS TRIGGER] High loss: {loss_val:.4f} > {BIRTH_THRESHOLD}")
                
                # Criterion 2: Validation-based trigger (if enabled)
                if USE_VALIDATION_TRIGGER and validation_accuracy is not None:
                    if validation_accuracy < VALIDATION_SPAWN_THRESHOLD:
                        should_spawn = True
                        print(f"  [NEUROGENESIS TRIGGER] Low validation accuracy: {validation_accuracy:.4f} < {VALIDATION_SPAWN_THRESHOLD}")
                
                # Criterion 3: Periodic check (every N epochs)
                if USE_VALIDATION_TRIGGER and epoch % VALIDATION_CHECK_FREQ == 0 and epoch > 0:
                    # In practice, we would compute validation accuracy here
                    # For now, we rely on the validation_accuracy parameter
                    pass
                
                if should_spawn:
                    expanded = self._expand_system()
                    if expanded:
                        # After expansion, the map is updated, so pick the new agent
                        idx = self.task_agent_map[self.current_task]
                    else:
                        # Could not expand (limit reached), use the routed agent
                        # But we should map this task to the routed agent for consistency
                        self.task_agent_map[self.current_task] = idx
        
        # 4. Train
        loss = self.agents[idx].adapt(x, y, self.current_task, is_new_task)
        
        self.usage[idx] += 1
        alpha = config.AGENT_LOSS_MA_ALPHA
        self.loss_ma[idx] = (1 - alpha) * self.loss_ma[idx] + alpha * loss
        return loss
    
    def consolidate_after_task(self):
        """
        Perform consolidation after training a task.
        This replays balanced samples from all seen classes to prevent forgetting.
        """
        if not CONSOLIDATION_AFTER_TASK:
            return 0.0
        
        print(f"  [CONSOLIDATION] Starting {CONSOLIDATION_EPOCHS} epochs of balanced replay...")
        
        total_consolidation_loss = 0.0
        num_batches = 0
        
        for epoch in range(CONSOLIDATION_EPOCHS):
            epoch_loss = 0.0
            batches_this_epoch = 0
            
            for _ in range(CONSOLIDATION_BATCHES_PER_EPOCH):
                # Try to consolidate each active agent
                for agent_idx in range(self.num_agents):
                    if self.usage[agent_idx] > 0:  # Agent has been used
                        loss = self.agents[agent_idx].consolidate_with_tiny_buffer(
                            epochs=1,
                            batch_size=getattr(config, 'TINY_BUFFER_BATCH_SIZE', 32)
                        )
                        if loss > 0:
                            epoch_loss += loss
                            batches_this_epoch += 1
            
            if batches_this_epoch > 0:
                avg_epoch_loss = epoch_loss / batches_this_epoch
                total_consolidation_loss += avg_epoch_loss
                num_batches += 1
                print(f"  [CONSOLIDATION] Epoch {epoch+1}/{CONSOLIDATION_EPOCHS}: avg loss = {avg_epoch_loss:.4f}")
        
        if num_batches > 0:
            avg_loss = total_consolidation_loss / num_batches
            print(f"  [CONSOLIDATION] Complete. Average loss: {avg_loss:.4f}")
            return avg_loss
        
        return 0.0

    def dream(self):
        # Replay old memories
        active_agents = [i for i in range(self.num_agents) if self.usage[i] > 10]
        if not active_agents: return

        original_task = self.current_task

        for _ in range(config.DREAM_BATCHES_PER_EPOCH):
            idx = active_agents[torch.randint(0, len(active_agents), (1,)).item()]
            batch = self.agents[idx].sample_memory(config.DREAM_BATCH_SIZE)
            if batch[0] is None: continue
            x_mem, y_mem, t_mem = batch

            dream_task_id = t_mem[0].item()
            self.current_task = dream_task_id
            
            # Refresh Agent
            loss = self.agents[idx].adapt(x_mem, y_mem, dream_task_id, is_new_task=False)
            
            # Refresh Router
            feats = self._get_node_features() 
            pred_rewards = self.routing_gnn(feats, self.agent_graph.adj)
            
            target = pred_rewards.clone().detach()
            # We enforce that the mapped agent is the correct one
            if dream_task_id in self.task_agent_map:
                correct_idx = self.task_agent_map[dream_task_id]
                target[correct_idx] = 10.0 # High reward
                # Punish others slightly
            else:
                target[idx] = -loss

            loss_gnn = F.mse_loss(pred_rewards, target)
            self.routing_opt.zero_grad(set_to_none=True)
            loss_gnn.backward()
            self.routing_opt.step()

        self.current_task = original_task


    def forward(self, x, memorize_steps=0):
        # 1. Check if we have a hard-coded expert for the current task context
        if self.current_task in self.task_agent_map:
            idx = self.task_agent_map[self.current_task]
        else:
            # Fallback to GNN routing if no expert assigned yet
            idx = self._route(train=False)
            
        return self.agents[idx](x, y=None, memorize_steps=memorize_steps)
