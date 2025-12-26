import time
import os
import glob
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from celery_app import process_pdf_summary, INBOX_DIR

class PDFHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.processed_files = set()  # Keep track of files we've already processed

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.process_file(event.src_path)

    def on_moved(self, event):
        # Handle files moved into the inbox as well
        if not event.is_directory and event.dest_path.lower().endswith('.pdf'):
            self.process_file(event.dest_path)

    def process_file(self, file_path):
        # Skip if we've already processed this file
        if file_path in self.processed_files:
            return
            
        # Ensure file is fully written before processing
        print(f"Detected new PDF: {os.path.basename(file_path)}. Waiting for file to be ready...")
        try:
            # Wait for file to be stable (not being written to)
            for attempt in range(20):  # Wait up to 10 seconds
                try:
                    size1 = os.path.getsize(file_path)
                    time.sleep(0.5)
                    size2 = os.path.getsize(file_path)
                    if size1 == size2 and size1 > 0:  # File is stable and not empty
                        break
                except (FileNotFoundError, PermissionError):
                    time.sleep(0.5)
            else:
                print(f"Warning: File {file_path} may still be copying...")
            
            print(f"Submitting PDF to Celery: {os.path.basename(file_path)}")
            result = process_pdf_summary.delay(file_path)
            print(f"Task submitted with ID: {result.id}")
            
            # Mark as processed
            self.processed_files.add(file_path)
            
        except Exception as e:
            print(f"Error submitting PDF {file_path}: {e}")

    def process_existing_files(self):
        """Process any PDF files that are already in the inbox"""
        print(f"Checking for existing PDF files in {INBOX_DIR}...")
        pdf_pattern = os.path.join(INBOX_DIR, "*.pdf")
        existing_pdfs = glob.glob(pdf_pattern)
        
        if existing_pdfs:
            print(f"Found {len(existing_pdfs)} existing PDF(s). Processing...")
            for pdf_file in existing_pdfs:
                print(f"Processing existing PDF: {os.path.basename(pdf_file)}")
                self.process_file(pdf_file)
                time.sleep(2)  # Small delay between files
        else:
            print("No existing PDF files found.")


def watch_inbox():
    """Main function to watch the inbox directory"""
    print("=" * 60)
    print("PDF Watcher Service")
    print("=" * 60)
    print(f"Monitoring directory: {INBOX_DIR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Ensure inbox directory exists
    if not os.path.exists(INBOX_DIR):
        print(f"Creating inbox directory: {INBOX_DIR}")
        os.makedirs(INBOX_DIR)
    
    # Create event handler and process existing files
    event_handler = PDFHandler()
    event_handler.process_existing_files()
    
    # Setup file watcher
    observer = Observer()
    observer.schedule(event_handler, INBOX_DIR, recursive=False)
    observer.start()
    
    print("")
    print("🔍 Watching for new PDF files...")
    print("📋 To add PDFs: Copy/move them to the inbox folder")
    print("⏹️  To stop: Press Ctrl+C")
    print("")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n📤 Stopping PDF watcher...")
        observer.stop()
        print("✅ PDF watcher stopped.")
    
    observer.join()

if __name__ == "__main__":
    watch_inbox()
