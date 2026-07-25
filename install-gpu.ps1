$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$uv = 'C:\Users\aleks_000\.local\bin\uv.exe'
if (-not (Test-Path -LiteralPath $uv)) {
  throw "uv not found: $uv"
}

$env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.python'
$env:UV_MANAGED_PYTHON = '1'

Write-Output "Installing Python, venv, and cache under $projectRoot"
& $uv python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Failed to install Python 3.12' }

& $uv venv --clear --managed-python --python 3.12 (Join-Path $projectRoot '.venv-gpu')
if ($LASTEXITCODE -ne 0) { throw 'Failed to create GPU environment' }

$python = Join-Path $projectRoot '.venv-gpu\Scripts\python.exe'
& $uv pip install --python $python torch==2.10.0 numpy --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'Failed to install PyTorch CUDA' }

& $python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
