@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-gpu\Scripts\python.exe" (
  echo GPU-окружение не найдено.
  echo Сначала установите PyTorch CUDA в .venv-gpu.
  pause
  exit /b 1
)

set "NODE_EXE=C:\Users\aleks_000\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
".venv-gpu\Scripts\python.exe" gpu-train-commander.py --node "%NODE_EXE%" --fresh --cycles 6 --battles 160 --epochs 6 --workers 16 --epsilon 0.16
pause
