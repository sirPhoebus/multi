@echo off
echo Starting Celery Worker for Windows...
echo Make sure Redis is running before proceeding!
echo.
pause
celery -A celery_app worker --loglevel=info --pool=solo
