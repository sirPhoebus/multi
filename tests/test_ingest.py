
from marl_scientist.knowledge.real_store import RealKnowledgeStore

def test_ingest():
    print("Testing Ingestion...")
    store = RealKnowledgeStore(persistence_path="test_kb.pkl")
    store.ingest_references("marl_scientist/ref.md")
    
    print(f"Loaded {len(store.documents)} documents.")
    if len(store.documents) > 20:
        print("PASS: Successfully parsed papers.")
    else:
        print("FAIL: Parsed too few papers.")

if __name__ == "__main__":
    test_ingest()
