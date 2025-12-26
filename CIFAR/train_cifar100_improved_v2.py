# train_cifar100_improved_v2.py
# Training script with all suggested improvements:
# 1. Tiny replay buffer per agent (10-20 images per class)
# 2. Improved birth criterion (lower threshold + validation trigger)
# 3. Increased SLOW_LR from 0.0001 to 0.001
# 4. Consolidation step after each task
# 5. Options for 5-task and 10-task versions

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import sys

# Import improved config and replace the default config module
import config_cifar100_improved_v2
sys.modules['config'] = config_cifar100_improved_v2
import config_cifar100_improved_v2 as config

from agent import DCCAgent
from orchestrator import GraphOrchestrator

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class SplitCIFAR100(Dataset):
    """Dataset for a specific task in Split CIFAR-100."""
    def __init__(self, task_id, train=True):
        """
        task_id: 0 to NUM_TASKS-1
        Each task gets NUM_CLASSES consecutive classes from CIFAR-100.
        """
        self.task_id = task_id
        self.train = train
        
        # CIFAR-100 normalization with improved transforms
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.RandomCrop(32, padding=4) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(config.CIFAR100_MEAN, config.CIFAR100_STD),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten to 3072
        ])
        
        # Download full CIFAR-100 dataset
        self.full_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=train, download=True, transform=transform
        )
        
        # Determine class indices for this task
        self.class_indices = list(range(task_id * config.NUM_CLASSES, (task_id + 1) * config.NUM_CLASSES))
        
        # Filter dataset to only include samples from these classes
        self.indices = [
            i for i, (_, label) in enumerate(self.full_dataset)
            if label in self.class_indices
        ]
        
        # Map original labels (0-99) to task-local labels (0 to NUM_CLASSES-1)
        self.label_map = {orig: local for local, orig in enumerate(self.class_indices)}
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        x, y = self.full_dataset[actual_idx]
        # Map label to local task label
        y_local = self.label_map[y]
        return x, y_local

def get_loader(task_id, train=True, batch_size=config.BATCH_SIZE):
    """Returns a DataLoader for a specific task."""
    dataset = SplitCIFAR100(task_id, train)
    
    use_multiprocessing = config.NUM_WORKERS > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2 if use_multiprocessing else None,
        persistent_workers=True if use_multiprocessing else False
    )

@torch.no_grad()
def evaluate(orchestrator, loader, task_id):
    """
    Evaluates the model on a specific task.
    CRITICAL: Switches the Orchestrator's context so it uses the correct Agent Map.
    """
    # 1. Save current state
    previous_task = orchestrator.current_task
    
    # 2. Switch Context (This allows the router to look up {TaskID -> AgentID})
    orchestrator.set_task(task_id)
    
    orchestrator.eval()
    correct, total = 0, 0
    
    for x, y in loader:
        x, y = x.to(config.DEVICE, non_blocking=True), y.to(config.DEVICE, non_blocking=True)
        
        # Forward pass (Router will check map or route dynamically)
        out = orchestrator(x, memorize_steps=config.MEMORIZE_STEPS)
        
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    
    orchestrator.train()
    
    # 3. Restore state
    orchestrator.set_task(previous_task)
    
    return correct / total

def apply_task_specific_lr(orchestrator, task_id):
    """Apply task-specific learning rate scaling if enabled."""
    if getattr(config, 'TASK_SPECIFIC_LR', False):
        lr_multiplier = getattr(config, 'INITIAL_LR_MULTIPLIER', 1.0) * \
                       (getattr(config, 'LR_DECAY_PER_TASK', 0.98) ** task_id)
        
        # Scale fast and slow learning rates
        for agent in orchestrator.agents:
            # Fast optimizers (list of optimizers for each level)
            for opt in agent.opt_fast:
                for param_group in opt.param_groups:
                    param_group['lr'] = config.FAST_LR * lr_multiplier
            # Slow optimizer (single optimizer)
            for param_group in agent.opt_slow.param_groups:
                param_group['lr'] = config.SLOW_LR * lr_multiplier
        
        if task_id % 5 == 0:  # Log every 5 tasks
            print(f"  Task {task_id}: Applied LR multiplier {lr_multiplier:.4f}")

def main():
    # Set random seed for reproducibility
    config.set_seed(config.SEED)
    
    print("=" * 80)
    print("DCC with All Suggested Improvements")
    print("=" * 80)
    print(f"Initializing DCC System on {config.DEVICE}")
    print(f"Config: Workers={config.NUM_WORKERS}, Compile={config.USE_COMPILE}")
    print(f"Experiment: Split CIFAR-100 ({config.NUM_TASKS} tasks, {config.NUM_CLASSES} classes each)")
    print(f"Input dimension: {config.INPUT_DIM}, Hidden: {config.HIDDEN_DIM}")
    print(f"TASK_MODE: {getattr(config, 'TASK_MODE', '20-task')}")
    print(f"Random seed: {config.SEED}")
    print()
    print("IMPROVEMENTS IMPLEMENTED:")
    print(f"1. Tiny replay buffer per agent: {config.TINY_BUFFER_SAMPLES_PER_CLASS} samples per class")
    print(f"2. Improved birth criterion: BIRTH_THRESHOLD={config.BIRTH_THRESHOLD}")
    print(f"   Validation trigger: {getattr(config, 'USE_VALIDATION_TRIGGER', False)}")
    print(f"3. Increased SLOW_LR: {config.SLOW_LR} (was 0.0001)")
    print(f"4. Consolidation after each task: {getattr(config, 'CONSOLIDATION_AFTER_TASK', False)}")
    print(f"5. Task configuration: {config.NUM_TASKS} tasks")
    print("=" * 80)
    
    # Start with 1 Initial Agent (It will grow!)
    agents = [DCCAgent().to(config.DEVICE) for _ in range(config.NUM_AGENTS)]
    
    if config.USE_COMPILE:
        print("Compiling Agents...")
        for ag in agents:
            ag.fast = torch.compile(ag.fast)
            ag.slow = torch.compile(ag.slow)
    else:
        print("Skipping torch.compile.")
    
    orchestrator = GraphOrchestrator(agents).to(config.DEVICE)
    results = []
    agent_assignments = {}  # Track task -> agent mapping
    task_losses = {i: [] for i in range(config.NUM_TASKS)}  # Track per-task losses
    
    for task_id in range(config.NUM_TASKS):
        print(f"\n{'='*60}")
        print(f"=== Training Task {task_id} ===")
        print(f"{'='*60}")
        orchestrator.set_task(task_id)
        
        # Apply task-specific learning rate if enabled
        apply_task_specific_lr(orchestrator, task_id)
        
        # REUSE MECHANISM: Check if we can reuse an existing agent
        if getattr(config, 'REUSE_ENABLED', False) and task_id > 0 and task_id not in orchestrator.task_agent_map:
            print(f"  [REUSE MECHANISM] Checking for agent reuse on Task {task_id}...")
            # Get a validation loader for the new task
            val_loader = get_loader(task_id, train=False, batch_size=config.BATCH_SIZE)
            reuse_agent = orchestrator.check_reuse_for_new_task(
                validation_loader=val_loader,
                task_id=task_id
            )
            if reuse_agent is not None:
                print(f"  [REUSE] Task {task_id} assigned to Agent {reuse_agent}")
        
        # Track agent assignment before training
        if task_id in orchestrator.task_agent_map:
            agent_id = orchestrator.task_agent_map[task_id]
            agent_assignments[task_id] = agent_id
            print(f"Task {task_id} assigned to Agent {agent_id} (reuse)")
        else:
            print(f"Task {task_id} not yet assigned to any agent")
        
        train_loader = get_loader(task_id, train=True)
        # Keep track of test loaders with their IDs
        test_loaders = [(t, get_loader(t, train=False, batch_size=512)) for t in range(task_id + 1)]
        
        # Use epochs per task from config
        epochs_per_task = config.EPOCHS_PER_TASK
        
        for epoch in range(epochs_per_task):
            # 1. Train Loop
            loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs_per_task}", leave=False)
            epoch_losses = []
            
            # Compute validation accuracy for neurogenesis trigger (every N epochs)
            validation_accuracy = None
            if getattr(config, 'USE_VALIDATION_TRIGGER', False) and epoch % getattr(config, 'VALIDATION_CHECK_FREQ', 5) == 0:
                # Evaluate on current task to get validation accuracy
                val_loader = get_loader(task_id, train=False, batch_size=512)
                validation_accuracy = evaluate(orchestrator, val_loader, task_id)
                print(f"  [VALIDATION] Task {task_id}, Epoch {epoch}: accuracy = {validation_accuracy:.4f}")
            
            for x, y in loop:
                x, y = x.to(config.DEVICE, non_blocking=True), y.to(config.DEVICE, non_blocking=True)
                
                # Adapt with improved neurogenesis criterion
                loss = orchestrator.adapt(
                    x, y, 
                    is_new_task=(epoch==0),
                    epoch=epoch,
                    validation_accuracy=validation_accuracy
                )
                epoch_losses.append(loss)
                
                loop.set_postfix(loss=f"{loss:.4f}")
            
            # Record average loss for this epoch
            avg_loss = np.mean(epoch_losses)
            task_losses[task_id].append(avg_loss)
            
            # 2. Dream Phase with frequency control
            dream_replay_freq = getattr(config, 'DREAM_REPLAY_FREQ', 1)
            if (epoch + 1) % dream_replay_freq == 0:
                orchestrator.dream()
        
        # 3. Consolidation after task (IMPROVEMENT #4)
        consolidation_loss = orchestrator.consolidate_after_task()
        if consolidation_loss > 0:
            print(f"  [CONSOLIDATION] Task {task_id} consolidation complete. Loss: {consolidation_loss:.4f}")
        
        # 4. Evaluation (Task-Aware)
        print(f"Evaluating on {len(test_loaders)} tasks...")
        accs = [evaluate(orchestrator, dl, t_id) for t_id, dl in test_loaders]
        
        results.append(accs)
        acc_str = ", ".join([f"Task {i}: {a:.4f}" for i, a in enumerate(accs)])
        print(f"Task {task_id} Complete. Accuracies: [{acc_str}]")
        
        # Track agent assignment after training
        if task_id in orchestrator.task_agent_map:
            agent_id = orchestrator.task_agent_map[task_id]
            agent_assignments[task_id] = agent_id
        
        # Print current agent assignments
        print(f"Current agent assignments: {agent_assignments}")
        
        # Check for transfer learning (agent reuse)
        if task_id > 0:
            current_agent = agent_assignments.get(task_id)
            previous_agents = [agent_assignments.get(t) for t in range(task_id) if t in agent_assignments]
            if current_agent in previous_agents:
                reused_task = [t for t, a in agent_assignments.items() if a == current_agent and t != task_id][0]
                print(f"TRANSFER LEARNING DETECTED: Task {task_id} reuses Agent {current_agent} from Task {reused_task}")
        
        # Print forgetting analysis
        if task_id > 0:
            print("Forgetting analysis:")
            for prev_task in range(task_id):
                prev_acc = results[task_id-1][prev_task] if len(results[task_id-1]) > prev_task else None
                curr_acc = accs[prev_task]
                if prev_acc is not None:
                    forgetting = prev_acc - curr_acc
                    trend = "↑" if forgetting < 0 else "↓" if forgetting > 0 else "→"
                    print(f"  Task {prev_task}: {prev_acc:.4f} -> {curr_acc:.4f} (forgetting: {forgetting:+.4f}) {trend}")
    
    # Plotting - Enhanced visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy over tasks
    ax1 = axes[0, 0]
    for t_id in range(config.NUM_TASKS):
        task_accs = [res[t_id] if len(res) > t_id else None for res in results]
        valid_points = [(i, a) for i, a in enumerate(task_accs) if a is not None]
        if valid_points:
            x_ax, y_ax = zip(*valid_points)
            ax1.plot(x_ax, y_ax, marker='o', label=f'Task {t_id}', linewidth=2, markersize=4)
            
    ax1.set_title(f"DCC Performance: Split CIFAR-100 (All Improvements)", fontsize=14)
    ax1.set_xlabel("Tasks Encountered", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(0.3, 1.05)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss per task
    ax2 = axes[0, 1]
    for t_id in range(config.NUM_TASKS):
        if task_losses[t_id]:
            ax2.plot(range(1, len(task_losses[t_id]) + 1), task_losses[t_id], 
                    label=f"Task {t_id}", linewidth=2)
    
    ax2.set_title("Training Loss per Task", fontsize=14)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Forgetting analysis
    ax3 = axes[1, 0]
    forgetting_matrix = np.zeros((config.NUM_TASKS, config.NUM_TASKS))
    for i in range(len(results)):
        for j in range(i + 1):
            if j < len(results[i]):
                forgetting_matrix[i, j] = results[i][j]
    
    im = ax3.imshow(forgetting_matrix, cmap='viridis', aspect='auto', vmin=0.3, vmax=1.0)
    ax3.set_title("Accuracy Matrix (Heatmap)", fontsize=14)
    ax3.set_xlabel("Task ID", fontsize=12)
    ax3.set_ylabel("Training Step", fontsize=12)
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Agent reuse visualization
    ax4 = axes[1, 1]
    if agent_assignments:
        agents = list(set(agent_assignments.values()))
        agent_colors = plt.cm.tab20(np.linspace(0, 1, len(agents)))
        
        for task, agent in sorted(agent_assignments.items()):
            color_idx = agents.index(agent)
            ax4.bar(task, 1, color=agent_colors[color_idx], edgecolor='black')
        
        ax4.set_title(f"Agent Reuse (Total: {len(agents)} agents)", fontsize=14)
        ax4.set_xlabel("Task ID", fontsize=12)
        ax4.set_ylabel("Agent Assignment", fontsize=12)
        ax4.set_xticks(range(config.NUM_TASKS))
        ax4.set_yticks([])
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=agent_colors[i], edgecolor='black', 
                                label=f'Agent {agents[i]}') for i in range(len(agents))]
        ax4.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    output_filename = f"dcc_cifar100_improved_v2_{config.NUM_TASKS}tasks.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    
    # Save detailed results
    results_filename = f"dcc_cifar100_results_improved_v2_{config.NUM_TASKS}tasks.txt"
    
    with open(results_filename, "w") as f:
        f.write(f"DCC with All Improvements - Results ({config.NUM_TASKS} tasks)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Config: TASK_MODE={getattr(config, 'TASK_MODE', '20-task')}\n")
        f.write(f"NUM_TASKS={config.NUM_TASKS}, NUM_CLASSES={config.NUM_CLASSES}\n")
        f.write(f"BIRTH_THRESHOLD={config.BIRTH_THRESHOLD}, SLOW_LR={config.SLOW_LR}\n")
        f.write(f"TINY_BUFFER_SAMPLES_PER_CLASS={config.TINY_BUFFER_SAMPLES_PER_CLASS}\n")
        f.write(f"CONSOLIDATION_AFTER_TASK={getattr(config, 'CONSOLIDATION_AFTER_TASK', False)}\n")
        f.write(f"Total agents spawned: {orchestrator.num_agents}\n")
        f.write(f"Agent assignments (task -> agent): {agent_assignments}\n")
        
        f.write("\nAccuracy matrix:\n")
        for i, accs in enumerate(results):
            f.write(f"After task {i}: {accs}\n")
        
        f.write("\nFinal Accuracies:\n")
        for t_id in range(config.NUM_TASKS):
            final_acc = results[-1][t_id] if len(results[-1]) > t_id else None
            if final_acc is not None:
                f.write(f"Task {t_id}: {final_acc:.4f}\n")
        
        # Calculate and report forgetting metrics
        f.write("\n=== Forgetting Analysis ===\n")
        if len(results) > 1:
            final_accs = results[-1]
            max_accs = [max(results[i][j] for i in range(j, len(results)) if j < len(results[i])) 
                       for j in range(len(final_accs))]
            
            forgetting_scores = [max_accs[j] - final_accs[j] for j in range(len(final_accs))]
            avg_forgetting = np.mean(forgetting_scores)
            max_forgetting = max(forgetting_scores)
            
            f.write(f"Average forgetting: {avg_forgetting:.4f}\n")
            f.write(f"Maximum forgetting: {max_forgetting:.4f}\n")
            f.write(f"Tasks with forgetting > 0.05: ")
            high_forgetting_tasks = [j for j, score in enumerate(forgetting_scores) if score > 0.05]
            f.write(f"{high_forgetting_tasks}\n")
            
            for j in high_forgetting_tasks:
                f.write(f"  Task {j}: forgetting={forgetting_scores[j]:.4f}, final_acc={final_accs[j]:.4f}\n")
        
        # Analyze transfer learning
        f.write("\n=== Transfer Learning Analysis ===\n")
        agent_to_tasks = {}
        for task, agent in agent_assignments.items():
            agent_to_tasks.setdefault(agent, []).append(task)
        
        transfer_count = 0
        for agent, tasks in agent_to_tasks.items():
            if len(tasks) > 1:
                transfer_count += len(tasks) - 1
                f.write(f"Agent {agent} handles {len(tasks)} tasks: {tasks}\n")
        
        f.write(f"\nTotal transfer learning instances: {transfer_count}\n")
        f.write(f"Agent reuse ratio: {transfer_count}/{config.NUM_TASKS-1} = {transfer_count/(config.NUM_TASKS-1):.2%}\n")
    
    print(f"\nDone! Results saved to {output_filename} and {results_filename}")
    print(f"Total agents spawned: {orchestrator.num_agents}")
    print(f"Agent assignments: {agent_assignments}")
    
    # Analyze transfer learning
    print("\n=== Transfer Learning Analysis ===")
    agent_to_tasks = {}
    for task, agent in agent_assignments.items():
        agent_to_tasks.setdefault(agent, []).append(task)
    
    transfer_count = 0
    for agent, tasks in agent_to_tasks.items():
        if len(tasks) > 1:
            transfer_count += len(tasks) - 1
            print(f"Agent {agent} handles {len(tasks)} tasks: {tasks}")
    
    print(f"Total transfer learning instances: {transfer_count}")
    print(f"Agent reuse ratio: {transfer_count}/{config.NUM_TASKS-1} = {transfer_count/(config.NUM_TASKS-1):.2%}")

if __name__ == "__main__":
    main()
