from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import pickle
from marl_scientist.core import ExperimentResult
from marl_scientist.llm.client import LLMClient

class RealKnowledgeStore:
    """
    Persistence-backed Knowledge Store using Local LLM Embeddings.
    """
    def __init__(self, persistence_path="kb_embeddings.pkl"):
        self.client = LLMClient()
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.persistence_path = persistence_path
        
        # Shared Journal (New discoveries)
        self.journal_documents: List[str] = []
        self.journal_metadatas: List[Dict] = []
        self.journal_embeddings: Optional[np.ndarray] = None
        
        self.load()

    def ingest_references(self, filepath: str):
        """
        Parses ref.md and ingests papers if not already loaded.
        """
        if len(self.documents) > 0:
            print(f"[KnowledgeStore] Loaded {len(self.documents)} papers (Dense).")
            return

        print(f"[KnowledgeStore] Ingesting papers from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Naive parsing of markdown list
        # Pattern 1: - [Title](link) - Focus: ... (Markdown link)
        # Pattern 2: Title - Link: ... - Focus: ... (Plain text)
        # We try a versatile regex or multiple
        
        # New robust parsing logic
        content_lines = content.split('\n')
        new_docs = []
        new_metas = []
        
        for line in content_lines:
            line = line.strip()
            if not line: continue
            
            # Remove leading dash/bullet if present
            clean_line = line.lstrip("- ").strip()
            
            # Check for Link and Focus
            if "Focus:" in clean_line:
                # Attempt to extract parts
                # 1. Look for Link
                link = "unknown"
                title = "unknown"
                focus = "unknown"
                
                # Split by " - " is risky if title has it, but standard format seems to be separators
                # Let's try regex for the specific format seen in file
                
                # Regex for: "Title (Venue) - Link: url - Focus: desc"
                # Optional "Authors: ... - "
                match = re.search(r"(.*?)\s-\s(?:Authors:.*?\s-\s)?(?:Link:|Link)\s*(.*?)\s-\sFocus:\s*(.*)", clean_line)
                
                if match:
                    title = match.group(1).strip()
                    link = match.group(2).strip()
                    focus = match.group(3).strip()
                    
                    # Clean title if it has markdown link syntax leftovers
                    # e.g. "[Title](...)"
                    m_link = re.search(r"\[(.*?)\]", title)
                    if m_link: 
                        title = m_link.group(1)
                        
                    new_docs.append(f"{title}. {focus}")
                    new_metas.append({"title": title, "link": link, "focus": focus, "source": "ref_md"})
                    continue
                    
                # Regex for Markdown Link style: "[Title](Link) - Focus: ..."
                match_md = re.search(r"\[(.*?)\]\((.*?)\)\s-\sFocus:\s*(.*)", clean_line)
                if match_md:
                     new_docs.append(f"{match_md.group(1)}. {match_md.group(3)}")
                     new_metas.append({"title": match_md.group(1), "link": match_md.group(2), "focus": match_md.group(3), "source": "ref_md"})
                     continue
        
        # Fallback if manual parsing loop finishes (we don't use the old regex findall anymore)
        matches = [] # Dummy to satisfy old code flow if we didn't replace it fully, but we will replace the loop below too.
        
        # Since we populated new_docs in the loop, we are good.
        pass
            
        if not new_docs:
            print("[KnowledgeStore] Warning: No papers found in ref.md")
            return

        self.documents.extend(new_docs)
        self.metadatas.extend(new_metas)
        
        # Generate Embeddings via LLM
        print(f"[KnowledgeStore] Generating embeddings for {len(new_docs)} papers via LLM. This may take a moment...")
        try:
            self.embeddings = self.client.get_embeddings_batch(self.documents)
            self.save()
        except Exception as e:
            print(f"[KnowledgeStore] Embedding generation failed: {e}")
            self.embeddings = np.zeros((len(self.documents), 768)) # Fallback dummy

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Semantic search using Dense Vector Cosine Similarity.
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
            
        try:
            query_vec = self.client.get_embedding(query).reshape(1, -1)
            
            # Search Static Knowledge
            sims = cosine_similarity(query_vec, self.embeddings).flatten()
            top_k_indices = sims.argsort()[-k:][::-1]
            
            hits = []
            for idx in top_k_indices:
                score = sims[idx]
                if score > 0.0: # Filter irrelevant
                    hits.append({
                        "text": self.documents[idx],
                        "metadata": self.metadatas[idx],
                        "distance": 1.0 - score, # Conversion for compatibility
                        "score": score,
                        "source": "local_shard"
                    })
            
            return hits
        except Exception as e:
            print(f"[KnowledgeStore] Search error: {e}")
            return []
            
    def add_journal_paper(self, paper: Dict):
        """
        Add a dynamic finding to the shared journal.
        """
        text = paper["text"]
        meta = paper["metadata"]
        
        self.journal_documents.append(text)
        self.journal_metadatas.append(meta)
        
        # Embed single new paper
        try:
            new_vec = self.client.get_embedding(text).reshape(1, -1)
            if self.journal_embeddings is None:
                self.journal_embeddings = new_vec
            else:
                self.journal_embeddings = np.vstack([self.journal_embeddings, new_vec])
                
            print(f"[KnowledgeStore] Publishing new paper: {meta['title']}")
            self.save()
        except Exception as e:
             print(f"[KnowledgeStore] Failed to publish paper: {e}")

    def search_journal(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.journal_embeddings is None:
            return []
            
        try:
            query_vec = self.client.get_embedding(query).reshape(1, -1)
            sims = cosine_similarity(query_vec, self.journal_embeddings).flatten()
            top_k_indices = sims.argsort()[-k:][::-1]
            
            hits = []
            for idx in top_k_indices:
                score = sims[idx]
                if score > 0.0:
                    hits.append({
                        "text": self.journal_documents[idx],
                        "metadata": self.journal_metadatas[idx],
                        "distance": 1.0 - score,
                        "source": "journal"
                    })
            return hits
        except Exception as e:
            return []

    def get_shard(self, agent_index: int, total_agents: int):
        """
        Returns a KnowledgeShard that has access to partial local docs
        BUT full access to the Journal (via parent pointer).
        """
        # Simple partitioning
        n_docs = len(self.documents)
        if n_docs == 0:
            return KnowledgeShard([], [], [], self.client, self)
            
        shard_size = n_docs // total_agents
        start = agent_index * shard_size
        end = start + shard_size if agent_index < total_agents - 1 else n_docs
        
        s_docs = self.documents[start:end]
        s_metas = self.metadatas[start:end]
        s_ids = list(range(start, end)) # virtual IDs
        
        # Note: We don't pass embeddings to shard, shard uses parent for search or we'd need to slice embeddings too.
        # For Simplicity, we will update KnowledgeShard to use parent's search logic restricted to indices
        # OR just let the shard have a subset.
        
        # Current Shard Search implementation in previous steps relied on `self.vectorizer`.
        # We need to update KnowledgeShard to use `self.parent_store` and filter by indices?
        # Actually, let's just give the shard its slice of embeddings.
        
        s_vecs = None
        if self.embeddings is not None:
             s_vecs = self.embeddings[start:end]
             
        return KnowledgeShard(s_docs, s_metas, s_vecs, self.client, self)

    def synthesize_new_paper(self, result: ExperimentResult, agent_id: str) -> Dict:
        """Create a paper dict from an experiment result."""
        title = f"Optimizing {result.config.algorithm}: {result.final_mean_reward:.0f} pts"
        focus = "Performance Improvement"
        text = f"{title}. Config: {result.config.hyperparameters}. Agent {agent_id} achieved {result.final_mean_reward} on CartPole."
        
        return {
            "text": text,
            "metadata": {
                "title": title,
                "focus": focus,
                "author": agent_id,
                "score": result.final_mean_reward
            }
        }

    def save(self):
        try:
            with open(self.persistence_path, 'wb') as f:
                pickle.dump({
                    "docs": self.documents,
                    "metas": self.metadatas,
                    "embs": self.embeddings,
                    "j_docs": self.journal_documents,
                    "j_metas": self.journal_metadatas,
                    "j_embs": self.journal_embeddings
                }, f)
        except Exception as e:
            print(f"[KnowledgeStore] Save failed: {e}")

    def suggest_config_from_paper(self, paper_text: str, paper_title: str = "Unknown Paper") -> Dict[str, Any]:
        """
        Uses LLM to extract a valid RL configuration from a paper summary.
        """
        if not paper_text or len(paper_text) < 10:
             # Skip empty papers
             with open("failed_papers.txt", "a", encoding="utf-8") as f:
                 f.write(f"SKIPPED (Empty): {paper_title}\n")
             return None
             
        prompt = f"""
You are an expert Machine Learning Engineer.
Read this research paper summary and extract a specific, valid hyperparameter configuration that helps achieve the goals mentioned.

Paper: "{paper_text}"

Output a JSON object with this schema:
{{
  "algorithm": "PPO" | "A2C" | "DQN" | "SAC",
  "hyperparameters": {{ ... }}
}}

Rules:
1. "algorithm" must be one of the allowed strings.
2. "hyperparameters" should include things like "learning_rate", "gamma", "ent_coef", etc. inferred from the text.
3. STRICT JSON only. No comments. No trailing commas.
4. Use standard float notation (e.g. 0.001), avoid unquoted expressions.
"""
        import json
        import re

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
                
                # Sanitization
                # 1. Strip <think>...</think> blocks if present (Reasoning Models)
                clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                
                # 2. Strip markdown code blocks
                clean_text = clean_response.replace("```json", "").replace("```", "").strip()
                
                # 3. Robust JSON extraction (find first { and last })
                start = clean_text.find("{")
                end = clean_text.rfind("}")
                if start != -1 and end != -1:
                    clean_text = clean_text[start:end+1]
                
                config = json.loads(clean_text)
                
                # Simple Validation
                if "algorithm" not in config or "hyperparameters" not in config:
                    raise ValueError("Missing fields in LLM output")
                    
                return config
            except Exception as e:
                print(f"[KnowledgeStore] Attempt {attempt+1}/{max_retries} failed: {e}")
                
                # Try simple repair for truncated JSON
                if "Expecting ',' delimiter" in str(e) or "Expecting value" in str(e) or "Unterminated string" in str(e):
                    try:
                        # Don't try to repair empty strings
                        if not clean_text:
                            raise ValueError("Empty response")
                            
                        # Heuristic: Append brackets and retry
                        print(f"[KnowledgeStore] Attempting repair on truncated JSON...")
                        repaired_text = clean_text + "}}"
                        config = json.loads(repaired_text)
                        
                        if "algorithm" in config and "hyperparameters" in config:
                            print(f"[KnowledgeStore] Repair successful!")
                            return config
                    except:
                        pass
                
                if attempt == max_retries - 1:
                    print(f"[DEBUG] Failed. Raw Response: {response[:200]}...")
                    # Log failure
                    with open("failed_papers.txt", "a", encoding="utf-8") as f:
                        f.write(f"FAILED (JSON): {paper_title} | Error: {e}\n")
                    return None
        return None

    def load(self):
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, 'rb') as f:
                data = pickle.load(f)
            self.documents = data.get("docs", [])
            self.metadatas = data.get("metas", [])
            self.embeddings = data.get("embs", None)
            self.journal_documents = data.get("j_docs", [])
            self.journal_metadatas = data.get("j_metas", [])
            self.journal_embeddings = data.get("j_embs", None)
            print(f"[KnowledgeStore] Loaded {len(self.documents)} papers (Dense).")
        except Exception as e:
            print(f"[KnowledgeStore] Load failed: {e}")

    def synthesize_new_paper(self, result: ExperimentResult, author_id: str) -> Dict[str, Any]:
        """
        Creates a 'paper' representation of an experiment result.
        """
        title = f"Empirical Study of {result.config.algorithm} with Reward {result.final_mean_reward:.1f}"
        
        # Summarize HPs
        hp_str = ", ".join([f"{k}={v}" for k,v in result.config.hyperparameters.items()])
        
        text = f"""
        Title: {title}
        Authors: {author_id}
        Abstract: We investigated {result.config.algorithm} with hyperparameters: {hp_str}.
        The experiment yielded a mean reward of {result.final_mean_reward:.2f}.
        This configuration showed {'high' if result.final_mean_reward > 400 else 'moderate'} stability.
        """
        
        paper = {
            "text": text.strip(),
            "metadata": {
                "title": title,
                "author": author_id,
                "reward": result.final_mean_reward,
                "source": "lab_generated"
            }
        }
        return paper

    def add_paper(self, paper: Dict[str, Any]):
        """
        Adds a single paper to the journal (dynamic knowledge).
        """
        self.journal_documents.append(paper["text"])
        self.journal_metadatas.append(paper["metadata"])
        # No embedding update for now to avoid latency, OR we do lazy embedding
        # Ideally we embed it:
        try:
             emb = self.client.get_embedding(paper["text"]).reshape(1, -1)
             if self.journal_embeddings is None:
                 self.journal_embeddings = emb
             else:
                 self.journal_embeddings = np.vstack([self.journal_embeddings, emb])
             print(f"[KnowledgeStore] Journal updated with paper: {paper['metadata']['title']}")
        except:
             pass

class KnowledgeShard:
    def __init__(self, docs, metas, embeddings, client, parent_store):
        self.documents = docs
        self.metadatas = metas
        self.embeddings = embeddings
        self.client = client
        self.parent_store = parent_store
        
    def synthesize_new_paper(self, result: ExperimentResult, author_id: str) -> Dict[str, Any]:
        return self.parent_store.synthesize_new_paper(result, author_id)

    def add_paper(self, paper: Dict[str, Any]):
        return self.parent_store.add_paper(paper)
        
    def search(self, query: str, k: int = 3):
        # 1. Local Search
        hits = []
        if self.embeddings is not None and len(self.embeddings) > 0:
            try:
                query_vec = self.client.get_embedding(query).reshape(1, -1)
                sims = cosine_similarity(query_vec, self.embeddings).flatten()
                top_indices = sims.argsort()[-k:][::-1]
                
                for idx in top_indices:
                    score = sims[idx]
                    if score > 0.0:
                        hits.append({
                            "text": self.documents[idx],
                            "metadata": self.metadatas[idx],
                            "distance": 1.0 - score,
                            "source": "local_shard"
                        })
            except:
                pass
                
        # 2. Journal Search
        journal_hits = self.parent_store.search_journal(query, k=k)
        hits.extend(journal_hits)
        
        # Sort
        hits.sort(key=lambda x: x.get("distance", 1.0))
        
        # Deduplicate
        seen = set()
        unique = []
        for h in hits:
            t = h["metadata"]["title"]
            if t not in seen:
                seen.add(t)
                unique.append(h)
                
        return unique[:k]
