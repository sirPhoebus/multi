#!/usr/bin/env python3
"""
Configuration script to set up .env file for the Celery LLM system
"""
import os
from dotenv import load_dotenv, set_key

def get_current_config():
    """Load and display current configuration"""
    load_dotenv()
    
    config = {
        'LLM_ENDPOINT_URL': os.getenv('LLM_ENDPOINT_URL', 'http://10.5.0.2:1234/v1/chat/completions'),
        'LLM_MODEL_NAME': os.getenv('LLM_MODEL_NAME', 'local-model'),
        'LLM_MAX_TOKENS': os.getenv('LLM_MAX_TOKENS', '1500'),
        'LLM_MAX_TOKENS_SUMMARY': os.getenv('LLM_MAX_TOKENS_SUMMARY', '1000'),
        'LLM_MAX_TOKENS_OPINION': os.getenv('LLM_MAX_TOKENS_OPINION', '1200'),
        'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'KNOWLEDGE_DIR': os.getenv('KNOWLEDGE_DIR', 'C:\\knowledge'),
        'INBOX_DIR': os.getenv('INBOX_DIR', 'C:\\inbox'),
        'PROCESSED_DIR': os.getenv('PROCESSED_DIR', 'C:\\inbox\\processed')
    }
    
    return config

def display_config(config):
    """Display current configuration"""
    print("Current Configuration:")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print()

def update_config():
    """Interactive configuration update"""
    env_file = ".env"
    config = get_current_config()
    
    print("Celery LLM System Configuration")
    print("=" * 50)
    print("Press Enter to keep current value, or type new value to change.")
    print()
    
    # LLM Endpoint URL
    current_url = config['LLM_ENDPOINT_URL']
    print(f"Current LLM Endpoint URL: {current_url}")
    new_url = input("Enter new LLM endpoint URL (or press Enter): ").strip()
    if new_url:
        set_key(env_file, 'LLM_ENDPOINT_URL', new_url)
        print(f"✅ Updated LLM_ENDPOINT_URL to: {new_url}")
    print()
    
    # Model Name
    current_model = config['LLM_MODEL_NAME']
    print(f"Current LLM Model Name: {current_model}")
    new_model = input("Enter new model name (or press Enter): ").strip()
    if new_model:
        set_key(env_file, 'LLM_MODEL_NAME', new_model)
        print(f"✅ Updated LLM_MODEL_NAME to: {new_model}")
    print()
    
    # Max Tokens - General
    current_tokens = config['LLM_MAX_TOKENS']
    print(f"Current Max Tokens (general): {current_tokens}")
    new_tokens = input("Enter new max tokens for general prompts (or press Enter): ").strip()
    if new_tokens and new_tokens.isdigit():
        set_key(env_file, 'LLM_MAX_TOKENS', new_tokens)
        print(f"✅ Updated LLM_MAX_TOKENS to: {new_tokens}")
    print()
    
    # Max Tokens - Summary
    current_summary_tokens = config['LLM_MAX_TOKENS_SUMMARY']
    print(f"Current Max Tokens (PDF summary): {current_summary_tokens}")
    new_summary_tokens = input("Enter new max tokens for PDF summaries (or press Enter): ").strip()
    if new_summary_tokens and new_summary_tokens.isdigit():
        set_key(env_file, 'LLM_MAX_TOKENS_SUMMARY', new_summary_tokens)
        print(f"✅ Updated LLM_MAX_TOKENS_SUMMARY to: {new_summary_tokens}")
    print()
    
    # Max Tokens - Opinion
    current_opinion_tokens = config['LLM_MAX_TOKENS_OPINION']
    print(f"Current Max Tokens (PDF opinion): {current_opinion_tokens}")
    new_opinion_tokens = input("Enter new max tokens for PDF opinions (or press Enter): ").strip()
    if new_opinion_tokens and new_opinion_tokens.isdigit():
        set_key(env_file, 'LLM_MAX_TOKENS_OPINION', new_opinion_tokens)
        print(f"✅ Updated LLM_MAX_TOKENS_OPINION to: {new_opinion_tokens}")
    print()
    
    # Redis URL
    current_redis = config['REDIS_URL']
    print(f"Current Redis URL: {current_redis}")
    new_redis = input("Enter new Redis URL (or press Enter): ").strip()
    if new_redis:
        set_key(env_file, 'REDIS_URL', new_redis)
        print(f"✅ Updated REDIS_URL to: {new_redis}")
    print()
    
    # Knowledge Directory
    current_knowledge = config['KNOWLEDGE_DIR']
    print(f"Current Knowledge Directory: {current_knowledge}")
    new_knowledge = input("Enter new knowledge directory path (or press Enter): ").strip()
    if new_knowledge:
        set_key(env_file, 'KNOWLEDGE_DIR', new_knowledge)
        print(f"✅ Updated KNOWLEDGE_DIR to: {new_knowledge}")
    print()
    
    # Inbox Directory
    current_inbox = config['INBOX_DIR']
    print(f"Current Inbox Directory: {current_inbox}")
    new_inbox = input("Enter new inbox directory path (or press Enter): ").strip()
    if new_inbox:
        set_key(env_file, 'INBOX_DIR', new_inbox)
        # Also update processed directory
        processed_dir = os.path.join(new_inbox, 'processed')
        set_key(env_file, 'PROCESSED_DIR', processed_dir)
        print(f"✅ Updated INBOX_DIR to: {new_inbox}")
        print(f"✅ Updated PROCESSED_DIR to: {processed_dir}")
    print()

def test_llm_connection():
    """Test connection to LLM endpoint"""
    import requests
    
    load_dotenv()
    url = os.getenv('LLM_ENDPOINT_URL')
    model = os.getenv('LLM_MODEL_NAME')
    
    print(f"Testing connection to: {url}")
    print(f"Using model: {model}")
    
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello, can you respond with just 'OK'?"}],
            "max_tokens": 10
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        print("✅ Connection successful!")
        print(f"LLM Response: {content}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    print("Celery LLM System - Configuration Tool")
    print("=" * 60)
    
    while True:
        print("\\nOptions:")
        print("1. View current configuration")
        print("2. Update configuration")
        print("3. Test LLM connection")
        print("4. Exit")
        
        choice = input("\\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            config = get_current_config()
            display_config(config)
            
        elif choice == '2':
            update_config()
            print("\\n✅ Configuration updated!")
            
        elif choice == '3':
            test_llm_connection()
            
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
