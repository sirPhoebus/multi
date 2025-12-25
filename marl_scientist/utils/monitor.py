import json
import os
import networkx as nx

def export_dashboard_data(agents, step, filepath="marl_scientist/dashboard/data.json"):
    """
    Exports the causal beliefs of all agents to a JSON file for the dashboard.
    """
    data = {
        "step": step,
        "agents": {}
    }
    
    # Sort agents by performance to determine rank
    sorted_agents = sorted(agents, key=lambda a: a.best_performance, reverse=True)
    rank_map = {a.agent_id: i+1 for i, a in enumerate(sorted_agents)}

    for agent in agents:
        # Get graph data from networkx
        if hasattr(agent, "causal_model"):
            g = agent.causal_model.graph
        else:
            # Fallback for Neural Agents or others without explicit text graphs
            g = nx.DiGraph()
        
        # Format nodes
        nodes = []
        for node in g.nodes():
            nodes.append({"id": node, "label": node})
            
        # Format edges
        edges = []
        for u, v, d in g.edges(data=True):
            edges.append({
                "from": u, 
                "to": v, 
                "label": f"{d.get('weight', 0):.2f}",
                "color": "green" if d.get('weight', 0) > 0 else "red",
                "width": abs(d.get('weight', 0)) * 5
            })
            
        data["agents"][agent.agent_id] = {
            "nodes": nodes,
            "edges": edges,
            "best_score": agent.best_performance,
            "rank": rank_map.get(agent.agent_id, 0),
            # Ideally we track efficiency too
        }
        
    # Ensure dir exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"[Dashboard] Data exported to {filepath}")
