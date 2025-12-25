from marl_scientist.llm.client import LLMClient
import numpy as np

def debug_llm():
    print("Testing connection to LM Studio...")
    client = LLMClient()
    
    if client.check_connection():
        print("SUCCESS: Connected to localhost:1234")
    else:
        print("FAILURE: Could not connect. Is LM Studio running and server started?")
        return

    print("Testing Embedding...")
    try:
        vec = client.get_embedding("Hello World")
        print(f"SUCCESS: Got embedding of shape {vec.shape}")
    except Exception as e:
        print(f"FAILURE: Embedding error: {e}")

if __name__ == "__main__":
    debug_llm()
