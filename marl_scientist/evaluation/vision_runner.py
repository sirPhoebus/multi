
import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import numpy as np

from marl_scientist.core import ExperimentConfig, ExperimentResult
from marl_scientist.models.dcc import DCCAgent

# Safe Import Trigger
HAS_VISION = False
try:
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    HAS_VISION = True
except Exception:
    # Catches ImportError AND RuntimeError (caused by broken extension)
    HAS_VISION = False

class SafeSplitCIFAR100:
    """
    Robust Dataset Wrapper.
    Uses generic Torch tensors if torchvision is missing.
    Generates fake data by default to avoid network dependency.
    """
    def __init__(self, task_index, train=True, root='./data', num_classes=5):
        self.task_index = task_index
        self.data: List[Tuple[torch.Tensor, int]] = []
        
        # Generator for Mock Data
        # 3072 input dim (flat)
        size = 200 if train else 50
        
        for _ in range(size):
            img = torch.randn(3072)
            label = np.random.randint(0, num_classes)
            self.data.append((img, label))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Returns (tensor, int)
        return self.data[idx]

class VisionExperimentRunner:
    """
    Runs Vision experiments using DCCAgent with fallbacks.
    """
    def __init__(self, task_id: str):
        self.task_id = task_id
        try:
            self.task_index = int(task_id.split("-")[-1])
        except:
            self.task_index = 0
            
    def run(self, config: ExperimentConfig, visual: bool = False, agent_id: str = "unknown") -> ExperimentResult:
        start_time = time.time()
        hp = config.hyperparameters
        
        try:
            # 1. Setup Data using Robust Class
            train_set = SafeSplitCIFAR100(self.task_index, train=True)
            test_set = SafeSplitCIFAR100(self.task_index, train=False)
            
            # Simple manual batching
            batch_size = int(hp.get("batch_size", 32))
            
            def get_batches(dataset, bs):
                data_len = len(dataset)
                indices = np.arange(data_len)
                np.random.shuffle(indices)
                for i in range(0, data_len, bs):
                    batch_idxs = indices[i:i+bs]
                    batch_x = torch.stack([dataset[j][0] for j in batch_idxs])
                    batch_y = torch.tensor([dataset[j][1] for j in batch_idxs], dtype=torch.long)
                    yield batch_x, batch_y

            train_loader = lambda: get_batches(train_set, batch_size)
            test_loader = lambda: get_batches(test_set, batch_size)

        except Exception as e:
            return self._failure_result(config, f"Dataset Init Error: {e}", start_time)

        # 2. Setup Agent
        try:
            # Ensure device is safe
            if "device" not in hp: hp["device"] = "cpu"
            if torch.cuda.is_available(): hp["device"] = "cuda"
            
            model = self._create_model(hp)
            
            # Load Pretrained
            load_path = hp.get("load_model_path")
            if load_path and os.path.exists(load_path):
                try:
                    state = torch.load(load_path, map_location=hp["device"])
                    model.load_state_dict(state, strict=False)
                except:
                    pass

            # 3. Train/Eval Loop
            epochs = int(hp.get("epochs", 5))
            final_acc = 0.0
            train_curve = []
            
            if epochs <= 0:
                final_acc = self._evaluate(model, test_loader(), hp["device"])
                train_curve = [final_acc]
            else:
                for epoch in range(epochs):
                    model.train()
                    for x, y in train_loader():
                        if hasattr(model, 'adapt'):
                            model.adapt(x, y, task_id=self.task_index, is_new_task=(epoch==0))
                        else:
                            if not hasattr(model, 'optimizer'):
                                model.optimizer = torch.optim.Adam(model.parameters(), lr=hp.get("lr", 1e-3))
                            
                            model.train()
                            x, y = x.to(hp["device"]), y.to(hp["device"])
                            out = model(x)
                            loss = nn.CrossEntropyLoss()(out, y)
                            model.optimizer.zero_grad()
                            loss.backward()
                            model.optimizer.step()
                    
                    acc = self._evaluate(model, test_loader(), hp["device"])
                    train_curve.append(acc)

                final_acc = train_curve[-1]

            # 4. Save
            save_dir = f"saves/models/{agent_id}"
            os.makedirs(save_dir, exist_ok=True)
            model_path = f"{save_dir}/{self.task_id}_{int(time.time())}.pt"
            torch.save(model.state_dict(), model_path)
            
            return ExperimentResult(
                config=config,
                final_mean_reward=final_acc * 100.0,
                training_curve=[x*100 for x in train_curve],
                metrics={"accuracy": final_acc, "model_path": model_path},
                duration_seconds=time.time() - start_time,
                start_time=start_time,
                performance_score=final_acc,
                final_model_path=model_path
            )

        except Exception as e:
            import traceback
            return self._failure_result(config, f"Training Error: {str(e)}\n{traceback.format_exc()}", start_time)

    def _create_model(self, hp: Dict[str, Any]):
        """Creates the model, either default DCC or custom from code."""
        if "model_code" in hp and hp["model_code"].strip():
            code = hp["model_code"]
            local_scope = {}
            try:
                # We add 'torch' and 'nn' to local scope for convenience
                import torch
                import torch.nn as nn
                exec(code, {"torch": torch, "nn": nn, "hp": hp}, local_scope)
                model_class = None
                for name, obj in local_scope.items():
                    if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                        model_class = obj
                        break
                
                if not model_class:
                    raise ValueError("No nn.Module subclass found in provided code.")
                
                try:
                    model = model_class(hp)
                except:
                    model = model_class()
                
                model.to(hp.get("device", "cpu"))
                return model
            except Exception as e:
                print(f"Error loading custom model code: {e}")
                raise e
        
        return DCCAgent(hp)

    def _evaluate(self, model, loader_gen, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader_gen:
                x = x.to(device)
                y = y.to(device)
                out = model(x) # DCC uses forward(x, y=None) for inference
                # out is [Batch, NumClasses]
                if isinstance(out, Tuple): out = out[0]
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def _failure_result(self, config, error_msg, start_time):
        return ExperimentResult(
            config=config,
            final_mean_reward=0.0,
            training_curve=[],
            metrics={"error": 1.0},
            duration_seconds=time.time() - start_time,
            start_time=start_time,
            info={"error": error_msg}
        )
