import os
import time
import shutil
import asyncio
import numpy as np
from marl_scientist.knowledge.watcher import KnowledgeWatcher
from marl_scientist.knowledge.real_store import RealKnowledgeStore

async def test_async_ingest_logic():
    # Setup paths
    test_dir = "test_knowledge_async"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    archive_dir = os.path.join(test_dir, "processed")
    
    print("[Test] Initialized test directories.")
    
    # 1. Initialize Store (with Mock LLM Client to avoid real calls)
    store = RealKnowledgeStore(persistence_path="test_async_kb.pkl")
    # Mocking client for speed
    class MockClient:
        async def async_get_embeddings_batch(self, texts):
            print(f"[MockLLM] Embedding {len(texts)} texts...")
            return np.zeros((len(texts), 768), dtype=np.float32)
            
    store.client = MockClient()
    
    # 2. Start Watcher
    watcher = KnowledgeWatcher(watch_dir=test_dir)
    watcher.start()
    
    # 3. Drop a file
    print("[Test] dropping file1.txt...")
    with open(os.path.join(test_dir, "file1.txt"), "w") as f:
        f.write("This is a test document.")
        
    await asyncio.sleep(2) # Give watcher time to see it
    
    # 4. Check Queue
    new_files = watcher.get_new_files()
    print(f"[Test] Watcher queue has {len(new_files)} files: {new_files}")
    assert len(new_files) >= 1, "Watcher failed to detect file1.txt"
    
    # 5. Process Queue
    await store.process_file_queue(new_files)
    
    # 6. Verify processing
    assert len(store.documents) == 1, "Store should have 1 document."
    assert "file1.txt" in os.listdir(archive_dir), "File should be moved to processed/"
    
    # 7. Test Caching (Same content, SAME filename/title)
    print("[Test] Recreating file1.txt (same content & title)...")
    with open(os.path.join(test_dir, "file1.txt"), "w") as f:
        f.write("This is a test document.")
        
    await asyncio.sleep(2)
    # Watcher should pick it up again
    new_files_2 = watcher.get_new_files()
    if not new_files_2: 
        print("[Test Warning] Watcher missed rapid recreate, forcing check...")
        new_files_2 = [os.path.join(test_dir, "file1.txt")]
        
    # Process again - should hit cache
    await store.process_file_queue(new_files_2)
    
    assert len(store.documents) == 2, "Store should have 2 documents."
    
    watcher.stop()
    print("[Test] PASSED.")
    
    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists("test_async_kb.pkl"):
        os.remove("test_async_kb.pkl")
    if os.path.exists("kb_cache.pkl"):
         os.remove("kb_cache.pkl")

if __name__ == "__main__":
    asyncio.run(test_async_ingest_logic())
