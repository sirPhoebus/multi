import logging
import sys
from rich.logging import RichHandler

def setup_logger(log_file="simulation.log"):
    """
    Configures a logger that writes to a file and the console (using Rich).
    """
    # Create a custom logger
    logger = logging.getLogger("MarlScientist")
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times if function is called repeatedly
    if logger.hasHandlers():
        return logger

    # 1. File Handler (Detailed)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 2. Console Handler (Rich)
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(logging.INFO)
    # RichHandler has its own formatter
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
