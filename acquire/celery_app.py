from celery import Celery
import requests
import json
from datetime import datetime
import os
import PyPDF2
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Celery app
app = Celery('llm_tasks')

# Celery configuration - Windows compatible settings
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_pool='solo',  # Use solo pool for Windows compatibility
    worker_concurrency=1,  # Single worker process
    task_always_eager=False,
    task_eager_propagates=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
)

# Configuration from environment variables
KNOWLEDGE_DIR = os.getenv('KNOWLEDGE_DIR', 'C:\\knowledge')
INBOX_DIR = os.getenv('INBOX_DIR', 'C:\\inbox')
PROCESSED_DIR = os.getenv('PROCESSED_DIR', 'C:\\inbox\\processed')
LLM_ENDPOINT_URL = os.getenv('LLM_ENDPOINT_URL', 'http://10.5.0.2:1234/v1/chat/completions')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'local-model')
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '1500'))
LLM_MAX_TOKENS_SUMMARY = int(os.getenv('LLM_MAX_TOKENS_SUMMARY', '1000'))
LLM_MAX_TOKENS_OPINION = int(os.getenv('LLM_MAX_TOKENS_OPINION', '1200'))

# Ensure directories exist

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(INBOX_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return None

def call_llm_api(prompt, max_tokens=None):
    """Helper function to call LM Studio API"""
    if max_tokens is None:
        max_tokens = LLM_MAX_TOKENS
    
    print(f"[DEBUG] Sending to LLM: max_tokens={max_tokens}, model={LLM_MODEL_NAME}")
    print(f"[DEBUG] Endpoint URL: {LLM_ENDPOINT_URL}")
    
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(LLM_ENDPOINT_URL, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    
    response_data = response.json()
    
    # Debug information
    usage_info = response_data.get('usage', {})
    finish_reason = response_data.get('choices', [{}])[0].get('finish_reason', 'unknown')
    actual_tokens = usage_info.get('completion_tokens', 'unknown')
    
    print(f"[DEBUG] LLM Response - finish_reason: {finish_reason}, completion_tokens: {actual_tokens}")
    
    if finish_reason == 'length':
        print(f"[WARNING] Response was truncated! Requested {max_tokens}, got {actual_tokens} tokens")
        print(f"[WARNING] Consider increasing max_tokens or check LLM server limits")
    
    return response_data.get('choices', [{}])[0].get('message', {}).get('content', '')

@app.task
def process_llm_prompt(prompt):
    """
    Task to send prompt to LM Studio and save response to knowledge folder
    """
    try:
        print(f"Sending prompt to LM Studio: {prompt[:100]}...")
        
        # Use the helper function to call LLM API
        llm_response = call_llm_api(prompt, max_tokens=LLM_MAX_TOKENS)
        
        if not llm_response:
            llm_response = "No response received from LLM"
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"llm_response_{timestamp}.txt"
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        
        # Write response to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Response:\n{llm_response}\n")
        
        print(f"Response saved to: {filepath}")
        return {"status": "success", "filename": filename, "response_length": len(llm_response)}
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Error contacting LM Studio: {str(e)}"
        print(error_msg)
        
        # Save error to file as well
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"llm_error_{timestamp}.txt"
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Error: {error_msg}\n")
        
        return {"status": "error", "error": error_msg, "filename": filename}
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return {"status": "error", "error": error_msg}

@app.task
def process_pdf_summary(pdf_path):
    """
    Task to extract text from PDF, summarize it using LLM, and save summary
    Returns the path to the summary file for chaining to opinion task
    """
    try:
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            raise Exception("Could not extract text from PDF")
        
        print(f"Extracted {len(pdf_text)} characters from PDF")
        
        # Create summarization prompt
        summary_prompt = f"""
Please provide a comprehensive summary of the following document. Include:
1. Main topic/subject
2. Key points and findings
3. Important conclusions or recommendations
4. Any significant data or statistics mentioned

Document content:
{pdf_text[:8000]}{'...' if len(pdf_text) > 8000 else ''}
"""
        
        # Get summary from LLM
        print("Requesting summary from LLM...")
        summary = call_llm_api(summary_prompt, max_tokens=LLM_MAX_TOKENS_SUMMARY)
        
        if not summary:
            summary = "No summary could be generated from the document."
        
        # Create summary filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        summary_filename = f"summary_{pdf_filename}_{timestamp}.txt"
        summary_filepath = os.path.join(KNOWLEDGE_DIR, summary_filename)
        
        # Write summary to file
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Source PDF: {os.path.basename(pdf_path)}\n")
            f.write(f"PDF Path: {pdf_path}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Summary:\n{summary}\n")
        
        print(f"Summary saved to: {summary_filepath}")
        
        # Move PDF to processed folder
        processed_pdf_path = os.path.join(PROCESSED_DIR, os.path.basename(pdf_path))
        shutil.move(pdf_path, processed_pdf_path)
        print(f"PDF moved to processed folder: {processed_pdf_path}")
        
        # Trigger opinion generation task
        print("Triggering opinion generation...")
        generate_opinion_on_summary.delay(summary_filepath, summary)
        
        return {
            "status": "success", 
            "pdf_file": os.path.basename(pdf_path),
            "summary_file": summary_filename,
            "summary_path": summary_filepath,
            "processed_pdf_path": processed_pdf_path,
            "summary_length": len(summary)
        }
        
    except Exception as e:
        error_msg = f"Error processing PDF {pdf_path}: {str(e)}"
        print(error_msg)
        
        # Save error to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0] if os.path.exists(pdf_path) else "unknown"
        error_filename = f"pdf_error_{pdf_filename}_{timestamp}.txt"
        error_filepath = os.path.join(KNOWLEDGE_DIR, error_filename)
        
        with open(error_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Source PDF: {pdf_path}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Error: {error_msg}\n")
        
        return {"status": "error", "error": error_msg, "error_file": error_filename}

@app.task
def generate_opinion_on_summary(summary_filepath, summary_content):
    """
    Task to generate an opinion on a summary using LLM
    """
    try:
        print(f"Generating opinion on summary: {summary_filepath}")
        
        # Create opinion prompt
        opinion_prompt = f"""
Please provide your expert opinion and analysis on the following document summary. Consider:
1. What are the strengths and weaknesses of the content?
2. What questions or concerns does this raise?
3. What additional information might be valuable?
4. What are the potential implications or significance of this content?
5. Any critical analysis or alternative perspectives?

Document Summary:
{summary_content}
"""
        
        # Get opinion from LLM
        print("Requesting opinion from LLM...")
        opinion = call_llm_api(opinion_prompt, max_tokens=LLM_MAX_TOKENS_OPINION)
        
        if not opinion:
            opinion = "No opinion could be generated for this summary."
        
        # Create opinion filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        summary_filename = os.path.splitext(os.path.basename(summary_filepath))[0]
        opinion_filename = f"opinion_{summary_filename}_{timestamp}.txt"
        opinion_filepath = os.path.join(KNOWLEDGE_DIR, opinion_filename)
        
        # Write opinion to file
        with open(opinion_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Source Summary: {os.path.basename(summary_filepath)}\n")
            f.write(f"Summary Path: {summary_filepath}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Opinion & Analysis:\n{opinion}\n")
        
        print(f"Opinion saved to: {opinion_filepath}")
        
        return {
            "status": "success",
            "summary_file": os.path.basename(summary_filepath),
            "opinion_file": opinion_filename,
            "opinion_path": opinion_filepath,
            "opinion_length": len(opinion)
        }
        
    except Exception as e:
        error_msg = f"Error generating opinion for {summary_filepath}: {str(e)}"
        print(error_msg)
        
        # Save error to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        error_filename = f"opinion_error_{timestamp}.txt"
        error_filepath = os.path.join(KNOWLEDGE_DIR, error_filename)
        
        with open(error_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Source Summary: {summary_filepath}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Error: {error_msg}\n")
        
        return {"status": "error", "error": error_msg, "error_file": error_filename}

if __name__ == '__main__':
    app.start()
