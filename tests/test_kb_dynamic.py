import os
import shutil
from marl_scientist.knowledge.real_store import RealKnowledgeStore, KnowledgeShard

def test_kb_dynamic():
    print("=== Testing Dynamic Knowledge Base ===")
    
    # 1. Clean up old test data
    if os.path.exists("test_kb.pkl"):
        os.remove("test_kb.pkl")
        
    # 2. Initialize Store
    kb = RealKnowledgeStore("test_kb.pkl")
    
    # 3. Add initial paper (Manually or via ingest)
    # We cheat and just add one manually to documents
    kb.documents = ["Initial study on PPO performance."]
    kb.metadatas = [{"title": "Init PPO", "focus": "performance", "link": "N/A"}]
    kb.ids = ["init_1"]
    kb.tfidf_matrix = kb.vectorizer.fit_transform(kb.documents)
    kb.save()
    
    # 4. Create two shards (Agent A and Agent B)
    # Both get the same init paper effectively
    shard_a = kb.get_shard(0, 2)
    shard_b = kb.get_shard(1, 2)
    
    print("Shards created.")
    
    # 5. Agent A 'publishes' a new paper
    print("Agent A publishing a new finding...")
    new_paper = {
        "id": "new_A_100",
        "text": "Title: Xenomorph Optimization. Focus: Advanced Aliens. Link: N/A",
        "metadata": {"title": "Xenomorph Optimization", "focus": "Aliens", "link": "N/A"}
    }
    shard_a.add_paper(new_paper)
    
    # 6. Verify Agent B can find it via Journal Search
    print("Agent B searching for 'Aliens'...")
    results = shard_b.search("Aliens", k=3)
    
    found = False
    for r in results:
        print(f"Result: {r['metadata']['title']} (Source: {r.get('source', 'unknown')})")
        if r['metadata']['title'] == "Xenomorph Optimization":
            found = True
            
    if found:
        print("SUCCESS: Agent B found the paper published by Agent A!")
    else:
        print("FAILURE: Agent B did not find the paper.")
        
    # 7. Verify Persistence
    print("Checking persistence...")
    kb2 = RealKnowledgeStore("test_kb.pkl")
    if len(kb2.journal_documents) > 0:
        print(f"SUCCESS: Journal persisted with {len(kb2.journal_documents)} papers.")
    else:
        print("FAILURE: Journal did not persist.")

if __name__ == "__main__":
    test_kb_dynamic()
