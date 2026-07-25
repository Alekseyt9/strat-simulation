@echo off
setlocal
cd /d "%~dp0"

start "" cmd /c "timeout /t 1 /nobreak >nul & start http://127.0.0.1:43131"

where py >nul 2>nul
if %errorlevel%==0 (
  py -m http.server 43131 --bind 127.0.0.1
  goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
  python -m http.server 43131 --bind 127.0.0.1
  goto :eof
)

echo.
echo Python не найден. Запустите любой локальный HTTP-сервер в этой папке.
pause
