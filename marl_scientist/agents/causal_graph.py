import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import pickle
import os

class CausalDiscoveryEngine:
    """
    Learns a Linear Structural Equation Model (SEM) from experiment history.
    
    Structure Tiers:
    0. Hyperparameters (Exogenous)
    1. Metrics (Mediators: e.g. Stability, Std Reward)
    2. Reward (Outcome)
    
    Assumptions:
    - No edges backwards (Reward -> HP is impossible).
    - No edges within Tier 0 (HPs are independent/randomized).
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.history: List[Dict[str, Any]] = []
        
        # Tiers definition
        # We will dynamically add nodes, but we know "reward" is tier 2
        self.tiers = {
            "reward": 2,
            "final_mean_reward": 2 # alias
        }
        
    def update_model(self, params: Dict[str, Any], metrics: Dict[str, Any], reward: float):
        """
        Ingest a new data point (Experiment Result).
        """
        entry = params.copy()
        entry.update(metrics)
        entry["reward"] = reward
        self.history.append(entry)
        
        # Update Tiers for new keys
        for k in params.keys():
            self.tiers[k] = 0 # Tier 0: Hyperparameters
            
        for k in metrics.keys():
            self.tiers[k] = 1 # Tier 1: Mediators
            
        # Re-learn graph if we have enough data
        if len(self.history) > 10 and len(self.history) % 5 == 0:
            self._learn_structure_and_parameters()
            
    def suggest_improvements(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest changes to HPs to maximize Reward, based on Total Causal Effect.
        """
        if len(self.history) < 10:
            return current_params # Not enough data to reason
            
        new_params = current_params.copy()
        
        # Calculate Total Effect of each HP on Reward
        # Sum of (path coefficient products)
        
        effects = {}
        for hp_node in self.graph.nodes():
            if self.tiers.get(hp_node, -1) != 0: continue
            if hp_node not in new_params: continue
            
            try:
                # Find all simple paths from HP to Reward
                paths = list(nx.all_simple_paths(self.graph, source=hp_node, target="reward"))
                total_effect = 0.0
                
                for path in paths:
                    path_effect = 1.0
                    for i in range(len(path)-1):
                        u, v = path[i], path[i+1]
                        weight = self.graph[u][v].get("weight", 0.0)
                        path_effect *= weight
                    total_effect += path_effect
                    
                effects[hp_node] = total_effect
            except:
                continue
                
        # Apply changes based on effects
        # If Effect > 0, we want to INCREASE the parameter
        # If Effect < 0, we want to DECREASE the parameter
        
        clean_effects = {k: round(float(v), 3) for k, v in effects.items()}
        print(f"  [Causal Inference] Estimated Effects: {clean_effects}")
        
        for hp, effect in effects.items():
            if abs(effect) < 0.05: continue # Ignore negligible effects
            
            val = new_params[hp]
            
            # Numeric params only
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                # Gradient ascent step
                step = 0.1 * np.sign(effect) 
                
                # Multiplicative update
                # If positive effect -> multiply by 1.1
                # If negative effect -> multiply by 0.9
                
                if effect > 0:
                    new_params[hp] = float(val) * 1.2
                else:
                    new_params[hp] = float(val) * 0.8
                    
        return new_params

    def _learn_structure_and_parameters(self):
        """
        Simplified PC-like algorithm + Linear Regression.
        """
        df = pd.DataFrame(self.history)
        
        # 1. Preprocessing
        # Drop non-numeric for now
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.dropna(axis=1, how='all') # Drop empty cols
        
        # Fill NaNs with mean
        numeric_df = numeric_df.fillna(numeric_df.mean())
        
        # Reset graph
        self.graph = nx.DiGraph()
        nodes = list(numeric_df.columns)
        self.graph.add_nodes_from(nodes)
        
        # 2. Structure Learning (Constraint-Based)
        # Iterate over all pairs (A, B) where Tier(A) < Tier(B)
        
        correlation_matrix = numeric_df.corr()
        
        for u in nodes:
            tier_u = self.tiers.get(u, -1)
            if tier_u == -1: continue # Unknown tier
            
            for v in nodes:
                tier_v = self.tiers.get(v, -1)
                if tier_v == -1: continue
                
                if tier_u < tier_v:
                    # Potential Edge U -> V
                    
                    # Check Marginal Correlation first
                    if u in correlation_matrix.columns and v in correlation_matrix.index:
                        corr = correlation_matrix.loc[v, u]
                        
                        # Threshold for existence (Weak PC)
                        if abs(corr) > 0.2:
                            self.graph.add_edge(u, v)
                            
        # 3. Parameter Learning (Coefficients)
        # For each node V, fit V ~ Parents(V)
        # We simplify: Just set weight = marginal correlation (approx for sparse graph)
        # in a full generic implementation we would run OLS for each node.
        
        # Improved: Run simple OLS for "reward" against its parents
        # (And parents against their parents... but let's stick to correlation weights for speed/stability in this prototype)
        
        for u, v in self.graph.edges():
            corr = correlation_matrix.loc[v, u]
            self.graph[u][v]["weight"] = corr
            
    def save_state(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Can't pickle networkx easily sometimes, but usually fine
        with open(filepath, 'wb') as f:
            pickle.dump({"history": self.history, "tiers": self.tiers}, f)
            
    def load_state(self, filepath: str):
        if not os.path.exists(filepath): return
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.history = data["history"]
            self.tiers = data.get("tiers", self.tiers)
            self._learn_structure_and_parameters()
        except:
            pass
