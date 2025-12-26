import time
import random
from datetime import datetime
from celery_app import process_llm_prompt

# Random prompts for the LLM
RANDOM_PROMPTS = [
    "Explain the concept of artificial intelligence in simple terms.",
    "What are the benefits and challenges of renewable energy?",
    "Describe the process of photosynthesis in plants.",
    "What is the importance of biodiversity in ecosystems?",
    "How does machine learning differ from traditional programming?",
    "Explain the water cycle and its impact on climate.",
    "What are the key principles of sustainable development?",
    "Describe the role of DNA in heredity.",
    "How do neural networks work in deep learning?",
    "What are the causes and effects of climate change?",
    "Explain the concept of quantum computing.",
    "What is the significance of the discovery of DNA structure?",
    "How do vaccines work to prevent diseases?",
    "Describe the process of evolution by natural selection.",
    "What are the main components of a computer system?",
    "Explain the greenhouse effect and global warming.",
    "What is the role of enzymes in biological processes?",
    "How does the internet work at a basic level?",
    "What are the principles of design thinking?",
    "Describe the structure and function of the human brain.",
    "What is blockchain technology and how does it work?",
    "Explain the concept of cryptocurrency.",
    "What are the benefits of exercise on mental health?",
    "How do plants adapt to their environments?",
    "What is the significance of the periodic table in chemistry?",
    "Describe the process of cellular respiration.",
    "What are the main types of renewable energy sources?",
    "How does GPS technology work?",
    "What is the importance of sleep for human health?",
    "Explain the concept of supply and demand in economics.",
]

def generate_random_prompt():
    """Generate a random prompt from the predefined list"""
    return random.choice(RANDOM_PROMPTS)

def main():
    """Main function to periodically generate and submit tasks"""
    print("Starting task generator...")
    print(f"Will submit tasks every 30 seconds to Celery")
    print("Press Ctrl+C to stop")
    
    task_count = 0
    
    try:
        while True:
            task_count += 1
            
            # Generate random prompt
            prompt = generate_random_prompt()
            
            # Submit task to Celery
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Task #{task_count}: Submitting prompt: '{prompt[:50]}...'")
            
            # Submit the task asynchronously
            result = process_llm_prompt.delay(prompt)
            
            print(f"[{timestamp}] Task #{task_count}: Submitted with ID: {result.id}")
            
            # Wait 30 seconds before next task
            print(f"[{timestamp}] Waiting 30 seconds for next task...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n\nTask generator stopped. Total tasks submitted: {task_count}")
        print("Goodbye!")

if __name__ == "__main__":
    main()
