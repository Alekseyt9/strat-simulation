@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if %errorlevel% neq 0 (
  echo Node.js не найден.
  pause
  exit /b 1
)

node diagnostic.js --matrix --trials 20 --units 150 --seed 1
pause
