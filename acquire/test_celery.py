"""
Test script to verify Celery configuration and LLM connectivity
"""
from dotenv import load_dotenv
load_dotenv()

from celery_app import process_llm_prompt
import time
import os

def test_celery_connection():
    """Test if Celery can submit and process a simple task"""
    print("Testing Celery configuration...")
    
    # Submit a test task
    test_prompt = "Say hello and confirm you're working properly."
    print(f"Submitting test prompt: {test_prompt}")
    
    try:
        # Submit task asynchronously
        result = process_llm_prompt.delay(test_prompt)
        print(f"Task submitted with ID: {result.id}")
        
        # Wait for result (with timeout)
        print("Waiting for result (timeout: 120 seconds)...")
        task_result = result.get(timeout=120)
        
        print("✅ Success! Task completed.")
        print(f"Result: {task_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_synchronous_llm():
    """Test LLM connectivity without Celery"""
    print("\nTesting direct LLM connectivity...")
    
    try:
        # Call the task function directly (synchronously)
        result = process_llm_prompt("Test prompt: What is 2+2?")
        print("✅ Direct LLM call successful!")
        print(f"Result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Direct LLM call failed: {e}")
        return False

if __name__ == "__main__":
    print("Celery LLM System Test")
    print("=" * 50)
    
    # Test 1: Direct LLM connectivity
    llm_ok = test_synchronous_llm()
    
    if llm_ok:
        print("\n" + "=" * 50)
        # Test 2: Celery task processing
        celery_ok = test_celery_connection()
        
        if celery_ok:
            print("\n🎉 All tests passed! System is ready.")
        else:
            print("\n⚠️  LLM works but Celery has issues. Check Redis and worker.")
    else:
        print("\n⚠️  LLM connection failed. Check LM Studio configuration.")
    
    print("\nTest complete.")
