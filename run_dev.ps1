$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python environment not found: $python"
}

# Preloaded MFA workers use roughly 340 MB each. MANDARIN_ASSESSMENT_CONCURRENCY defaults to 2.
# Ready workers align concurrently; shared model loading and the CLI fallback remain locked.
# Remove this line to run `mfa align_one` for every request instead.
$env:TONE_MFA_WARM = "1"

& $python -m uvicorn main:app `
    --host 127.0.0.1 `
    --port 8011 `
    --reload `
    --reload-dir $PSScriptRoot `
    --reload-exclude "outputs" `
    --reload-exclude "*.wav" `
    --reload-exclude "*.mp3"
