import torch
import os
import platform
import random
import numpy as np

# ================================================================
# SEED FOR REPRODUCIBILITY
# ================================================================
SEED = 42  # Default seed for reproducible runs

def set_seed(seed=SEED):
    """
    Set random seeds for reproducibility.
    Call this function at the beginning of your training script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set random seed to {seed} for reproducibility")

# System
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Windows compatibility
USE_COMPILE = (
    True 
    and torch.cuda.is_available() 
    and os.name != 'nt' 
    and platform.system() != 'Windows'
)

# Architecture - CIFAR-100: 3x32x32 = 3072 input dimensions
NUM_AGENTS = 1
INPUT_DIM = 3072  # 3*32*32
HIDDEN_DIM = 512  # Increased for more complex RGB images
NUM_CLASSES = 5   # Each task has 5 classes

# ================================================================
# TASK SETTINGS - Choose one configuration
# ================================================================
# Option 1: 20 tasks (full CIFAR-100, 5 classes per task)
# Option 2: 10 tasks (half CIFAR-100, 10 classes per task) 
# Option 3: 5 tasks (quarter CIFAR-100, 20 classes per task)
TASK_MODE = "10-task"  # Options: "20-task", "10-task", "5-task"

if TASK_MODE == "20-task":
    NUM_TASKS = 20
    NUM_CLASSES = 5  # 5 classes per task
elif TASK_MODE == "10-task":
    NUM_TASKS = 10
    NUM_CLASSES = 10  # 10 classes per task
elif TASK_MODE == "5-task":
    NUM_TASKS = 5
    NUM_CLASSES = 20  # 20 classes per task
else:
    raise ValueError(f"Unknown TASK_MODE: {TASK_MODE}")

print(f"Using {TASK_MODE} configuration: {NUM_TASKS} tasks, {NUM_CLASSES} classes per task")

# Nested Learning - Optimized for better performance
FAST_FREQS = [0.98, 0.95]  # More frequent updates
SLOW_FREQS = [0.80]  # More frequent slow updates
FAST_LR = 0.001  # Reduced learning rate for stability

# IMPROVEMENT #3: Increased SLOW_LR from 0.0001 to 0.001
SLOW_LR = 0.001  # Increased slow learning rate (was 0.0001)

FAST_WEIGHT = 0.7  # Increased fast weight for quicker adaptation
SLOW_WEIGHT = 0.3  # Decreased slow weight

# Memory & Gate - Enhanced for better consolidation
SURPRISE_THRESHOLD = 0.003  # More sensitive to surprise
MEMORIZE_STEPS = 5  # Increased test-time adaptation
BUFFER_CAPACITY = 8000  # Increased buffer for better memory
CONSOLIDATION_BATCH_SIZE = 96  # Larger consolidation batches

# IMPROVEMENT #1: Tiny replay buffer per agent (DER++ style)
TINY_BUFFER_SAMPLES_PER_CLASS = 18  # 10-20 images per class as suggested
TINY_BUFFER_BATCH_SIZE = 32  # Batch size for tiny buffer replay
CONSOLIDATION_LR = 1e-3  # Learning rate for consolidation

# Routing - Fine-tuned based on results
AGENT_SELECTION_TOP_K = 4  # Balanced selection
AGENT_LOSS_MA_ALPHA = 0.08  # Slightly slower moving average
ROUTING_EPSILON = 0.08  # Less exploration over time
USAGE_PENALTY_WEIGHT = 0.8  # Higher penalty to prevent overuse

# Sleeping / Dreaming - Enhanced for better forgetting prevention
DREAM_BATCHES_PER_EPOCH = 80  # Increased dream replay
DREAM_BATCH_SIZE = 48  # Larger dream batches
DREAM_REPLAY_FREQ = 2  # Replay every 2 epochs

# Training
BATCH_SIZE = 128
EPOCHS_PER_TASK = 25  # Increased from 20 for better learning
TOTAL_EPOCHS = EPOCHS_PER_TASK * NUM_TASKS

# IMPROVEMENT #2: Improved birth criterion
# Lower threshold + validation-based trigger
BIRTH_THRESHOLD = 2.0  # Reduced from 4.0 (spawn earlier)
USE_VALIDATION_TRIGGER = True  # Enable validation-based spawning
VALIDATION_SPAWN_THRESHOLD = 0.3  # Spawn if validation accuracy < 30%
VALIDATION_CHECK_FREQ = 5  # Check validation every N epochs

# REUSE MECHANISM: Test existing agents before spawning new ones
REUSE_ENABLED = True  # Enable agent reuse mechanism
REUSE_ZERO_SHOT_THRESHOLD = 0.28  # Reuse agent if it gets >45% accuracy zero-shot (no adaptation)
REUSE_ADAPT_THRESHOLD = 0.58  # Reuse agent if it gets >35% accuracy after 1-2 epochs of adaptation
REUSE_TEST_EPOCHS = 4  # Test for 1-2 epochs on validation set (if zero-shot fails)

# IMPROVEMENT #4: Consolidation step after each task
CONSOLIDATION_AFTER_TASK = True  # Enable consolidation after each task
CONSOLIDATION_EPOCHS = 3  # 1-2 epochs of balanced replay
CONSOLIDATION_BATCHES_PER_EPOCH = 20  # Number of consolidation batches per epoch

# Regularization - Enhanced to prevent overfitting
WEIGHT_DECAY = 2e-4  # Increased weight decay
GRADIENT_CLIP = 0.5  # Tighter gradient clipping
DROPOUT_RATE = 0.1  # Added dropout for regularization

# Agent limit in orchestrator
MAX_AGENTS = 12  # Reduced from 20 to force more reuse

# Task-specific adjustments
TASK_SPECIFIC_LR = True  # Enable task-specific learning rate scaling
INITIAL_LR_MULTIPLIER = 1.0  # Start with base LR
LR_DECAY_PER_TASK = 0.98  # Decay LR slightly per new task

# CIFAR-100 specific
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Debug and logging
VERBOSE = True
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQ = 5  # Save checkpoint every N tasks
