$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python environment not found: $python"
}

& $python -m uvicorn main:app `
    --host 127.0.0.1 `
    --port 8011 `
    --reload `
    --reload-dir $PSScriptRoot `
    --reload-exclude "outputs" `
    --reload-exclude "*.wav" `
    --reload-exclude "*.mp3"
