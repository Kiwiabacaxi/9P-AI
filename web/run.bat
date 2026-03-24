@echo off
setlocal

set PORT=8080

echo » compilando...
cd server
go build -o mlp-server.exe .
if errorlevel 1 (
    echo ERRO: build falhou
    exit /b 1
)
cd ..

echo » encerrando processo anterior na porta %PORT%...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    taskkill /PID %%p /F >nul 2>&1
)

echo » iniciando servidor em http://localhost:%PORT%
start "" server\mlp-server.exe

echo » aguardando servidor...
:wait_loop
timeout /t 1 /nobreak >nul
curl -s http://localhost:%PORT%/api/status >nul 2>&1
if errorlevel 1 goto wait_loop

echo » abrindo navegador...
start http://localhost:%PORT%

echo » servidor rodando. Pressione Ctrl+C para encerrar.
pause >nul
