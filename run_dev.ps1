$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = Join-Path $PSScriptRoot "env\Scripts\python.exe" }
if (-not (Test-Path $python)) {
    throw "Python environment not found: $python"
}

# The assessor runs separately on loopback port 8011; this website uses port 8012.
# The public website does not preload its own MFA worker.
if (-not $env:MANDARIN_PUBLIC_MODE) { $env:MANDARIN_PUBLIC_MODE = "1" }
if ($env:MANDARIN_PUBLIC_MODE -eq "1") {
    $port = 8012
    $env:TONE_MFA_WARM = "0"
    if (-not $env:MANDARIN_ASSESSOR_URL) { $env:MANDARIN_ASSESSOR_URL = "http://127.0.0.1:8011" }
    if (-not $env:MANDARIN_PRACTICE_DATA_DIR) { $env:MANDARIN_PRACTICE_DATA_DIR = Join-Path $PSScriptRoot "practice_data\public" }
    if (-not $env:MANDARIN_USER_DATA_DIR) { $env:MANDARIN_USER_DATA_DIR = Join-Path $PSScriptRoot "practice_data\public\archive" }
    if (-not $env:GOOGLE_CLIENT_ID -and -not $env:MANDARIN_DEV_AUTH) { $env:MANDARIN_DEV_AUTH = "1" }
}
else { $env:TONE_MFA_WARM = "1"; $port = 8011 }

& $python -m uvicorn main:app `
    --host 127.0.0.1 `
    --port $port `
    --reload `
    --reload-dir $PSScriptRoot `
    --reload-exclude "outputs" `
    --reload-exclude "*.wav" `
    --reload-exclude "*.mp3"
