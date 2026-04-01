@echo off
echo Starting 9P-AI...
echo.

echo [1/2] Building frontend...
cd /d %~dp0web\frontend && call npm run build
if errorlevel 1 (
    echo ERRO: Build do frontend falhou!
    pause
    exit /b 1
)
echo Build concluido.
echo.

echo [2/2] Starting Go server -> http://localhost:8080
echo Press Ctrl+C to stop.
echo.
cd /d %~dp0web\server && go run main.go
