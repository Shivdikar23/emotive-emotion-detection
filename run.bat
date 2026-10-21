@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
  echo The project virtual environment is missing. Create it first:
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
".venv\Scripts\streamlit.exe" run app.py
