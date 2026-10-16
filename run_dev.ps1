$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python environment not found: $python"
}

# A pre-loaded MFA aligner (about 340 MB resident) cuts an assessment from ~6 s to ~2 s with identical scores.
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
