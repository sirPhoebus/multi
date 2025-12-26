@echo off
echo Starting PDF Watcher...
echo Make sure Celery Worker is running in another terminal!
echo The watcher will monitor C:\inbox for new PDF files.
echo.
pause
python pdf_watcher_enhanced.py
