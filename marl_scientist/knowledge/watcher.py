from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os
import queue
import logging

class KnowledgeEventHandler(FileSystemEventHandler):
    """Handles file creation events in the knowledge directory."""
    def __init__(self, processing_queue: queue.Queue):
        self.queue = processing_queue
        
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.txt') or event.src_path.endswith('.md'):
            # self.queue.put(event.src_path)
            # Use small delay to ensure file write is complete?
            # Better: put it in the queue, consumer handles reading logic
            print(f"[Watcher] Detected new file: {event.src_path}")
            self.queue.put(event.src_path)

    def on_moved(self, event):
        # If user renames a file into the dir or within
        if not event.is_directory and (event.dest_path.endswith('.txt') or event.dest_path.endswith('.md')):
             print(f"[Watcher] Detected moved file: {event.dest_path}")
             self.queue.put(event.dest_path)

class KnowledgeWatcher:
    """
    Watches a directory for new knowledge files and queues them 
    for processing by the main agent loop.
    """
    def __init__(self, watch_dir: str):
        self.watch_dir = watch_dir
        self.queue = queue.Queue()
        self.event_handler = KnowledgeEventHandler(self.queue)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.watch_dir, recursive=False)
        self.is_running = False
        
    def start(self):
        if not os.path.exists(self.watch_dir):
            os.makedirs(self.watch_dir, exist_ok=True)
            
        self.observer.start()
        self.is_running = True
        print(f"[KnowledgeWatcher] Monitoring {self.watch_dir} for new inputs...")
        
    def stop(self):
        self.observer.stop()
        self.observer.join()
        self.is_running = False
        print("[KnowledgeWatcher] Stopped.")
        
    def get_new_files(self) -> list:
        """Returns all currently queued files."""
        files = []
        try:
            while True:
                files.append(self.queue.get_nowait())
        except queue.Empty:
            pass
        return files
