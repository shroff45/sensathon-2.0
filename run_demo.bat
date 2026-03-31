@echo off
echo ============================================
echo   Cross-Layer Physics-Based Vehicle IDS
echo   Starting Demo...
echo ============================================
echo.

echo [1/2] Starting Dashboard...
start "Dashboard" cmd /k "streamlit run dashboard.py"
timeout /t 3 /nobreak >nul

echo [2/2] Starting Demo (Software Mode)...
python quick_demo.py

echo.
echo Demo complete! Check the dashboard.
pause
