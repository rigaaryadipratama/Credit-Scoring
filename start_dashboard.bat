@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Buat virtual environment dan install requirements.txt terlebih dahulu. Lihat README.md.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
