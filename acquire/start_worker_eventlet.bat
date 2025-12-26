@echo off
echo Installing eventlet for better Windows compatibility...
pip install eventlet
echo.
echo Starting Celery Worker with eventlet...
echo Make sure Redis is running before proceeding!
echo.
pause
celery -A celery_app worker --loglevel=info --pool=eventlet --concurrency=1
