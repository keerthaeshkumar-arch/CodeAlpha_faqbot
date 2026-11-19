@echo off
title Pink FAQ Chatbot Launcher
color 0D
echo ========================================================
echo             Starting Pink FAQ Chatbot...
echo ========================================================
echo.
echo Application will be available at:
echo http://localhost:8501
echo.
echo Press Ctrl+C in this command window to stop the server.
echo ========================================================
echo.

cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501

pause
