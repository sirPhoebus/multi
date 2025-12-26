import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from celery_app import process_pdf_summary, INBOX_DIR

class PDFHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.process_file(event.src_path)

    def on_moved(self, event):
        # Handle files moved into the inbox as well
        if not event.is_directory and event.dest_path.lower().endswith('.pdf'):
            self.process_file(event.dest_path)

    def process_file(self, file_path):
        # Ensure file is fully written before processing (simple retry)
        print(f"Detected new PDF: {file_path}. Waiting for file to be ready...")
        try:
            for _ in range(10):
                try:
                    size1 = os.path.getsize(file_path)
                    time.sleep(0.5)
                    size2 = os.path.getsize(file_path)
                    if size1 == size2:
                        break
                except FileNotFoundError:
                    time.sleep(0.5)
            print(f"Submitting PDF to Celery: {file_path}")
            process_pdf_summary.delay(file_path)
        except Exception as e:
            print(f"Error submitting PDF {file_path}: {e}")


def watch_inbox():
    print(f"Watching folder for PDFs: {INBOX_DIR}")
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, INBOX_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    watch_inbox()

