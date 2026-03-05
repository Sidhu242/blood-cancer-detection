@echo off
REM Activate the virtual environment and start the Flask app
cd /d "%~dp0"
REM Use localhost for offline compatibility
start "" http://127.0.0.1:5000/
"cancer_env\Scripts\python.exe" app.py
