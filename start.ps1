# ═══════════════════════════════════════════════
#  Sorbot AI Trading System — Start Script
#  Kills processes on ports 80, 8000, 8081
#  then launches Docker Compose
# ═══════════════════════════════════════════════

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SORBOT AI TRADING SYSTEM" -ForegroundColor Cyan
Write-Host "  Starting up..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Kill processes on required ports ──
Write-Host "[1/3] Freeing ports..." -ForegroundColor Yellow

$ports = @(80, 8000, 8081)
foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        $pid = $conn.OwningProcess
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "  Killing $($proc.ProcessName) on port $port (PID: $pid)" -ForegroundColor Red
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
}
Write-Host "  Ports 80, 8000, 8081 are free." -ForegroundColor Green
Write-Host ""

# ── Build and start Docker containers ──
Write-Host "[2/3] Building Docker images..." -ForegroundColor Yellow
docker compose build

Write-Host ""
Write-Host "[3/3] Starting containers..." -ForegroundColor Yellow
docker compose up -d

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  SORBOT IS RUNNING!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend:   http://localhost" -ForegroundColor White
Write-Host "  Backend:    http://localhost:8081" -ForegroundColor White
Write-Host "  AI Engine:  http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "  Logs:       docker compose logs -f" -ForegroundColor DarkGray
Write-Host "  Stop:       docker compose down" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
