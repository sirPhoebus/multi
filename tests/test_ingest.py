
from marl_scientist.knowledge.real_store import RealKnowledgeStore

def test_ingest():
    print("Testing Ingestion...")
    import os
    os.makedirs("test_knowledge", exist_ok=True)
    with open("test_knowledge/test_paper.txt", "w") as f:
        f.write("This is a test summary about reinforcement learning.")
        
    store = RealKnowledgeStore(persistence_path="test_kb.pkl")
    store.ingest_folder("test_knowledge")
    
    print(f"Loaded {len(store.documents)} documents.")
    if len(store.documents) > 0:
        print("PASS: Successfully parsed folder.")
    else:
        print("FAIL: Parsed no documents.")

if __name__ == "__main__":
    test_ingest()
