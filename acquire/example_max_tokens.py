#!/usr/bin/env python3
"""
Example script demonstrating configurable max_tokens settings
"""
from dotenv import load_dotenv
load_dotenv()

import os
from celery_app import (
    LLM_MAX_TOKENS, 
    LLM_MAX_TOKENS_SUMMARY, 
    LLM_MAX_TOKENS_OPINION,
    call_llm_api
)

def demonstrate_max_tokens():
    """Show how different max_tokens work"""
    print("Max Tokens Configuration Demo")
    print("=" * 50)
    
    print(f"General prompts max_tokens: {LLM_MAX_TOKENS}")
    print(f"PDF summary max_tokens: {LLM_MAX_TOKENS_SUMMARY}")
    print(f"PDF opinion max_tokens: {LLM_MAX_TOKENS_OPINION}")
    print()
    
    # Show the environment variables
    print("Environment Variables:")
    print(f"  LLM_MAX_TOKENS = {os.getenv('LLM_MAX_TOKENS', 'Not set')}")
    print(f"  LLM_MAX_TOKENS_SUMMARY = {os.getenv('LLM_MAX_TOKENS_SUMMARY', 'Not set')}")
    print(f"  LLM_MAX_TOKENS_OPINION = {os.getenv('LLM_MAX_TOKENS_OPINION', 'Not set')}")
    print()
    
    print("Usage examples:")
    print("  call_llm_api('prompt')  # Uses LLM_MAX_TOKENS")
    print("  call_llm_api('prompt', LLM_MAX_TOKENS_SUMMARY)  # Uses summary max_tokens")
    print("  call_llm_api('prompt', LLM_MAX_TOKENS_OPINION)  # Uses opinion max_tokens")
    print("  call_llm_api('prompt', 500)  # Uses custom max_tokens")
    print()
    
    print("To change these values:")
    print("1. Edit the .env file directly")
    print("2. Run: python configure.py")

if __name__ == "__main__":
    demonstrate_max_tokens()
