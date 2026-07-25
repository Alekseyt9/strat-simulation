@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPYCACHEPREFIX=F:\src\strat\.python-cache"
set "PIP_CACHE_DIR=F:\src\strat\.pip-cache"
set "NODE_EXE=C:\Users\aleks_000\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
".venv-gpu\Scripts\python.exe" train-commander-v4.py --node "%NODE_EXE%" %*
