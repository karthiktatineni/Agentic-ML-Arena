@echo off
title AutoML Arena Launcher
echo ========================================================
echo        Starting AutoML Arena (Backend + Frontend)
echo ========================================================
echo.

cd /d "%~dp0"

:: Start FastAPI Backend
echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "AutoML Arena - Backend" cmd /k "cd /d ""%~dp0"" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload"

:: Start Next.js Frontend
echo [2/2] Starting Next.js Frontend on http://localhost:3000 ...
start "AutoML Arena - Frontend" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"

echo.
echo ========================================================
echo  AutoML Arena is launching!
echo  - Backend API: http://localhost:8000
echo  - Swagger UI:  http://localhost:8000/docs
echo  - Frontend UI: http://localhost:3000
echo ========================================================
echo.
