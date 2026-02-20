@echo off
REM ═══════════════════════════════════════════════
REM  Sorbot AI Trading System — Start Script
REM  Kills processes on ports 80, 8000, 8081
REM  then launches Docker Compose
REM ═══════════════════════════════════════════════

echo.
echo ========================================
echo   SORBOT AI TRADING SYSTEM
echo   Starting up...
echo ========================================
echo.

REM ── Kill processes on required ports ──
echo [1/3] Freeing ports...

for %%P in (80 8000 8081) do (
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr :%%P ^| findstr LISTENING 2^>nul') do (
        echo   Killing process on port %%P (PID: %%A)
        taskkill /F /PID %%A >nul 2>&1
    )
)
echo   Ports 80, 8000, 8081 are free.
echo.

REM ── Build and start Docker containers ──
echo [2/3] Building Docker images...
docker compose build

echo.
echo [3/3] Starting containers...
docker compose up -d

echo.
echo ========================================
echo   SORBOT IS RUNNING!
echo ========================================
echo.
echo   Frontend:   http://localhost
echo   Backend:    http://localhost:8081
echo   AI Engine:  http://localhost:8000
echo.
echo   Logs:       docker compose logs -f
echo   Stop:       docker compose down
echo ========================================
echo.

pause
