@echo off
echo Starting 9P-AI servers...
echo.
echo [1] Go backend  -> http://localhost:8080
echo [2] Vite frontend -> http://localhost:5173
echo.
echo Press Ctrl+C to stop both.
echo.

start "9P-AI Backend" cmd /c "cd /d %~dp0web\server && go run main.go"
cd /d %~dp0web\frontend && npm run dev
