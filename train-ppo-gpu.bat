@echo off
setlocal
cd /d "%~dp0"
set "UV_CACHE_DIR=F:\src\strat\.uv-cache"
set "NODE_EXE=C:\Users\aleks_000\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
".venv-gpu\Scripts\python.exe" train-ppo-commander.py --node "%NODE_EXE%" --cycles 8 --battles 192 --validation-battles 96 --workers 16
