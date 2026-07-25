@echo off
setlocal
cd /d "%~dp0"
set "NODE_EXE=C:\Users\aleks_000\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
".venv-gpu\Scripts\python.exe" train-commander-v3.py --node "%NODE_EXE%" --cycles 10 --battles 192 --validation-battles 120 --workers 16
pause
