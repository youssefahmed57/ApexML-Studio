@echo off
cd /d "%~dp0"
echo ================================================================
echo ApexML Final - Simple Pro
echo ApexML UI ^> Safe Training ^> Predict 10 Cases
echo ================================================================

python -c "import streamlit, pandas, sklearn" >nul 2>&1
if errorlevel 1 (
  echo First run: installing required packages...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    echo Run: python doctor.py
    pause
    exit /b 1
  )
)

echo Starting on the first available Streamlit port...
python launch.py
pause
