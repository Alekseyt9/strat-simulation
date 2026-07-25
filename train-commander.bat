@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if %errorlevel% neq 0 (
  echo Node.js не найден.
  pause
  exit /b 1
)

node train-commander.js --generations 12 --population 16 --battles 16 --sigma 0.16
pause
