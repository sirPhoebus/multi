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
        
        # [NEW] Embedding Cache
        self.embedding_cache: Dict[int, np.ndarray] = {}
        self.load_cache()
        
        self.load()

    def load_cache(self, path="kb_cache.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.embedding_cache = pickle.load(f)

    def save_cache(self, path="kb_cache.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.embedding_cache, f)

    def process_file_queue(self, file_paths: List[str]):
        """
        Processes a specific list of files (e.g. from the Watcher).
        Hashes content to check cache before embedding.
        """
        if not file_paths:
            return

        import shutil
        archive_path = os.path.join(os.path.dirname(file_paths[0]), "processed")
        os.makedirs(archive_path, exist_ok=True)
        
        new_docs = []
        new_metas = []
        processed_files = []
        docs_to_embed = []
        doc_indices_to_embed = [] # Map index in new_docs to index in docs_to_embed
        
        cached_embeddings = []

        print(f"[KnowledgeStore] Processing {len(file_paths)} queued files...")

        for file_path in file_paths:
            if not os.path.exists(file_path): continue
            
            try:
                filename = os.path.basename(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    shutil.move(file_path, os.path.join(archive_path, filename))
                    continue
                
                title = os.path.splitext(filename)[0].replace("_", " ").title()
                full_text = f"{title}. {content}"
                text_hash = hash(full_text)
                
                new_docs.append(full_text)
                new_metas.append({
                    "title": title, 
                    "filename": filename,
                    "source": "filesystem",
                    "ingested_at": str(os.path.getmtime(file_path))
                })
                processed_files.append((file_path, filename))
                
                # Check Cache
                if text_hash in self.embedding_cache:
                    cached_embeddings.append(self.embedding_cache[text_hash])
                else:
                    docs_to_embed.append(full_text)
                    doc_indices_to_embed.append(len(new_docs) - 1)
                    # We will insert placeholders or logic to merge later
                    
            except Exception as e:
                print(f"[KnowledgeStore] Failed to read {file_path}: {e}")

        if not new_docs:
            return

        # Generate Embeddings for novel items
        final_new_embeddings = np.zeros((len(new_docs), 768), dtype=np.float32)
        
        # Fill from cache first (inefficient double loop but clear)
        # Re-iterating to map correctly
        current_embed_idx = 0
        current_cache_idx = 0
        
        new_vectors_computed = []

        if docs_to_embed:
             print(f"[KnowledgeStore] Embedding {len(docs_to_embed)} new items (Cache Hit: {len(new_docs)-len(docs_to_embed)})...")
             computed_vectors = self.client.get_embeddings_batch(docs_to_embed)
             
             # Save to cache
             for txt, vec in zip(docs_to_embed, computed_vectors):
                 self.embedding_cache[hash(txt)] = vec
             self.save_cache()
             
             new_vectors_computed = computed_vectors
        
        # Assemble final array
        # This logic is a bit tricky with split lists. 
        # Easier: Re-loop using cache which is now complete.
        
        assembled_list = []
        for doc in new_docs:
            assembled_list.append(self.embedding_cache[hash(doc)])
            
        final_new_embeddings = np.array(assembled_list, dtype=np.float32)

        # Add to memory
        self.documents.extend(new_docs)
        self.metadatas.extend(new_metas)
        
        if self.embeddings is None:
            self.embeddings = final_new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, final_new_embeddings])
        
        # Save DB
        self.save()
        
        # Move Files
        for src, fname in processed_files:
            try:
                shutil.move(src, os.path.join(archive_path, fname))
            except:
                pass
            
        print(f"[KnowledgeStore] Ingested {len(new_docs)} items.")

    def ingest_folder(self, folder_path: str):
        """
        Reads all text files in a directory. Redirects to process_file_queue.
        """
        if not os.path.exists(folder_path): return
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt") or f.endswith(".md")]
        if files:
            self.process_file_queue(files)

    def ingest_references(self, filepath: str):
        """
        Deprecated: Parses ref.md. Redirects to ingest_folder if filepath is a directory.
        """
        if os.path.isdir(filepath):
            return self.ingest_folder(filepath)
        
        # Original ref.md logic (left for compatibility if needed, but updated to use ingest_folder style)
        if not os.path.exists(filepath): return
        
        print(f"[KnowledgeStore] Ingesting papers from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_docs = []
        new_metas = []
        
        content_lines = content.split('\n')
        for line in content_lines:
            line = line.strip()
            if not line: continue
            clean_line = line.lstrip("- ").strip()
            
            if "Focus:" in clean_line:
                match = re.search(r"(.*?)\s-\s(?:Authors:.*?\s-\s)?(?:Link:|Link)\s*(.*?)\s-\sFocus:\s*(.*)", clean_line)
                if match:
                    title = match.group(1).strip()
                    new_docs.append(f"{title}. {match.group(3).strip()}")
                    new_metas.append({"title": title, "source": "ref_md"})
                    continue
                
                match_md = re.search(r"\[(.*?)\]\((.*?)\)\s-\sFocus:\s*(.*)", clean_line)
                if match_md:
                     new_docs.append(f"{match_md.group(1)}. {match_md.group(3).strip()}")
                     new_metas.append({"title": match_md.group(1), "source": "ref_md"})

        if not new_docs: return

        self.documents.extend(new_docs)
        self.metadatas.extend(new_metas)
        self.embeddings = self.client.get_embeddings_batch(self.documents)
        self.save()

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
                        "embedding": self.journal_embeddings[idx], # [NEW] Return the vector
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
        text = f"{title}. Config: {result.config.hyperparameters}. Agent {agent_id} achieved {result.final_mean_reward} on {result.config.env_id}."
        
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

    def suggest_config_from_paper(self, paper_text: str, paper_title: str = "Unknown Paper", negative_knowledge: str = "") -> Dict[str, Any]:
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
{negative_knowledge}

Output a JSON object with this schema:
{{
  "algorithm": "PPO" | "A2C" | "DQN" | "SAC",
  "hyperparameters": {{ ... }},
  "env_id": "CartPole-v1" | "Acrobot-v1" | "Pendulum-v1" | "LunarLander-v3"
}}

Rules:
1. "algorithm" must be one of the allowed strings.
2. "env_id" should be inferred from the text if possible (e.g. if it mentions balance/poles, use CartPole; if it mentions landing/moon, use LunarLander). Default to "CartPole-v1" if unsure.
3. "hyperparameters" should include things like "learning_rate", "gamma", "ent_coef", etc. inferred from the text.
4. STRICT JSON only. No comments. No trailing commas.
5. Provide your response in English. No Chinese.
"""
        import json
        import re

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat_completion([{"role": "user", "content": prompt}], temperature=0.2)
                
                if not response:
                    raise ValueError("Empty response from LLM")

                # Sanitization
                # 1. Strip <think>...</think> blocks if present
                # Handle truncated think blocks (where </think> is missing)
                clean_response = response
                if "<think>" in response:
                    if "</think>" in response:
                        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                    else:
                        # Truncated inside think. Remove everything from <think> to end
                        start_think = response.find("<think>")
                        clean_response = response[:start_think].strip()

                # 2. Strip markdown code blocks
                clean_text = clean_response.replace("```json", "").replace("```", "").strip()
                
                # 3. Robust JSON extraction (find first { and last })
                start = clean_text.find("{")
                end = clean_text.rfind("}")
                if start != -1 and end != -1:
                    clean_text = clean_text[start:end+1]
                
                if not clean_text:
                    raise ValueError("No JSON object found in response")

                config = json.loads(clean_text)
                
                # Simple Validation
                if "algorithm" not in config or "hyperparameters" not in config:
                    raise ValueError("Missing fields in LLM output JSON")
                    
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
