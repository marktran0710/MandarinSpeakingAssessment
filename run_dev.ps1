$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = Join-Path $PSScriptRoot "env\Scripts\python.exe" }
if (-not (Test-Path $python)) {
    throw "Python environment not found: $python"
}

# Preloaded MFA workers use roughly 340 MB each. MANDARIN_ASSESSMENT_CONCURRENCY defaults to 2.
# Ready workers align concurrently; shared model loading and the CLI fallback remain locked.
# Remove this line to run `mfa align_one` for every request instead.
$env:TONE_MFA_WARM = "1"
if (-not $env:MANDARIN_PUBLIC_MODE) { $env:MANDARIN_PUBLIC_MODE = "1" }
if ($env:MANDARIN_PUBLIC_MODE -eq "1") {
    if (-not $env:MANDARIN_PRACTICE_DATA_DIR) { $env:MANDARIN_PRACTICE_DATA_DIR = Join-Path $PSScriptRoot "practice_data\public" }
    if (-not $env:MANDARIN_USER_DATA_DIR) { $env:MANDARIN_USER_DATA_DIR = Join-Path $PSScriptRoot "practice_data\public\archive" }
    if (-not $env:GOOGLE_CLIENT_ID -and -not $env:MANDARIN_DEV_AUTH) { $env:MANDARIN_DEV_AUTH = "1" }
}

& $python -m uvicorn main:app `
    --host 127.0.0.1 `
    --port 8011 `
    --reload `
    --reload-dir $PSScriptRoot `
    --reload-exclude "outputs" `
    --reload-exclude "*.wav" `
    --reload-exclude "*.mp3"
