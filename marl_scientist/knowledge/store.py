from typing import List, Dict
import dataclasses

@dataclasses.dataclass
class Paper:
    title: str
    content_summary: str
    embedding: List[float]

class KnowledgeStore:
    """
    Interface for the Vector Database (Papers).
    """
    def __init__(self):
        self.papers: Dict[str, Paper] = {}
        
    def add_paper(self, paper: Paper):
        self.papers[paper.title] = paper
        
    def search(self, query_embedding: List[float], k: int = 5) -> List[Paper]:
        # Mock search: return random/all papers
        return list(self.papers.values())[:k]

    def synthesize_new_paper(self, experiment_results: List) -> Paper:
        """
        Mocks the generation of a new paper from results.
        """
        return Paper(
            title="Automated Discovery Reuslt",
            content_summary=f"Found a config with reward {experiment_results[0].final_mean_reward}",
            embedding=[0.0] * 64
        )
