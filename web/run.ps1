$PORT = 8080

Write-Host "» compilando..."
Push-Location server
go build -o mlp-server.exe .
if ($LASTEXITCODE -ne 0) { Write-Error "Build falhou"; exit 1 }
Pop-Location

Write-Host "» encerrando processo anterior na porta $PORT..."
Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

Write-Host "» iniciando servidor em http://localhost:$PORT"
$server = Start-Process -FilePath "server\mlp-server.exe" -PassThru

Write-Host "» aguardando servidor..."
do {
    Start-Sleep -Milliseconds 200
    $ready = try { (Invoke-WebRequest "http://localhost:$PORT/api/status" -UseBasicParsing -TimeoutSec 1).StatusCode -eq 200 } catch { $false }
} while (-not $ready)

Write-Host "» abrindo navegador..."
Start-Process "http://localhost:$PORT"

Write-Host "» servidor rodando (PID $($server.Id)). Pressione Enter para encerrar."
Read-Host | Out-Null
Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
