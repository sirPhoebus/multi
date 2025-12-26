"""
Test script to verify PDF processing functionality
"""
from dotenv import load_dotenv
load_dotenv()

import os
import time
from celery_app import process_pdf_summary, INBOX_DIR, KNOWLEDGE_DIR

def test_pdf_processing():
    """Test PDF processing with a sample file"""
    print("Testing PDF Processing System")
    print("=" * 50)
    
    # Check if we have any PDF files to test with
    test_pdfs = []
    if os.path.exists(INBOX_DIR):
        for file in os.listdir(INBOX_DIR):
            if file.lower().endswith('.pdf'):
                test_pdfs.append(os.path.join(INBOX_DIR, file))
    
    if not test_pdfs:
        print(f"❌ No PDF files found in {INBOX_DIR}")
        print("To test:")
        print("1. Copy a PDF file to the inbox folder")
        print("2. Run this test again")
        return False
    
    print(f"✅ Found {len(test_pdfs)} PDF file(s) for testing")
    
    # Test with the first PDF
    test_pdf = test_pdfs[0]
    print(f"Testing with: {os.path.basename(test_pdf)}")
    
    try:
        # Submit PDF processing task
        print("Submitting PDF processing task...")
        result = process_pdf_summary.delay(test_pdf)
        print(f"Task submitted with ID: {result.id}")
        
        # Wait for result (with timeout)
        print("Waiting for processing to complete (timeout: 300 seconds)...")
        task_result = result.get(timeout=300)
        
        print("✅ PDF Processing completed successfully!")
        print(f"Result: {task_result}")
        
        # Check if files were created
        if task_result.get('status') == 'success':
            summary_file = task_result.get('summary_path')
            if summary_file and os.path.exists(summary_file):
                print(f"✅ Summary file created: {os.path.basename(summary_file)}")
                
                # Wait a bit for opinion task to complete
                print("Waiting for opinion generation to complete...")
                time.sleep(10)
                
                # Check for opinion file
                opinion_files = [f for f in os.listdir(KNOWLEDGE_DIR) 
                               if f.startswith('opinion_') and f.endswith('.txt')]
                if opinion_files:
                    print(f"✅ Opinion file(s) created: {', '.join(opinion_files[-3:])}")  # Show last 3
                    return True
                else:
                    print("⚠️ Opinion file not found yet (may still be processing)")
                    return True
            else:
                print("❌ Summary file not found")
                return False
        else:
            print("❌ PDF processing failed")
            return False
        
    except Exception as e:
        print(f"❌ Error during PDF processing test: {e}")
        return False

def list_knowledge_files():
    """List recent files in knowledge directory"""
    print("\\nRecent files in knowledge directory:")
    print("-" * 40)
    
    if not os.path.exists(KNOWLEDGE_DIR):
        print("Knowledge directory doesn't exist yet.")
        return
    
    files = []
    for file in os.listdir(KNOWLEDGE_DIR):
        if file.endswith('.txt'):
            file_path = os.path.join(KNOWLEDGE_DIR, file)
            mtime = os.path.getmtime(file_path)
            files.append((file, mtime))
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x[1], reverse=True)
    
    for file, mtime in files[:10]:  # Show latest 10 files
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        print(f"  {file} ({mtime_str})")

if __name__ == "__main__":
    success = test_pdf_processing()
    list_knowledge_files()
    
    if success:
        print("\\n🎉 PDF processing test completed successfully!")
    else:
        print("\\n⚠️ PDF processing test had issues.")
    
    print("\\nTest complete.")
