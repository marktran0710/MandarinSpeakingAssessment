# Taiwan Mandarin AI

A web application for practicing Taiwanese Mandarin tones, built with FastAPI and an integrated browser interface. Learners choose a text, record their voice, and receive syllable and phrase feedback, pitch charts, and practice history.

The system checks recordings against a **known transcript**: audio preparation → voice calibration → syllable alignment with Montreal Forced Aligner (MFA) → F0 extraction with Praat → tone and tone-sandhi verification. The current assessment flow does not use automatic speech recognition (ASR).

## Features

- Chapter 5 word and phrase practice, textbook sentence practice, and custom text practice.
- Personalized pitch-range calibration using five reference tones.
- `Good`, `Okay`, or `Needs practice` feedback. Insufficient acoustic evidence results in `Okay`.
- English, Traditional Chinese, Vietnamese, Indonesian, and Hindi interfaces.
- Learner profiles, recording history, feedback ratings, and an administration page.

The interface defaults to `easy` for both learner groups (`taiwanese` and `foreigner`). Evaluation tools can still select `standard` or `easy` through the API's `level` field.

## Quick start

Open PowerShell in the project root and run:

```powershell
.\run_dev.ps1
```

- Application: <http://127.0.0.1:8011>
- Administration: <http://127.0.0.1:8011/admin>
- API documentation: <http://127.0.0.1:8011/docs>

The script uses `env\Scripts\python.exe`, enables automatic source reloading, and sets `TONE_MFA_WARM=1` to preload the aligner. Run commands from the project root so relative paths resolve correctly.

To run without automatic reloading:

```powershell
$env:TONE_MFA_WARM = "1"
.\env\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8011
```

## New environment setup

The working environment uses **Python 3.12**:

```powershell
py -3.12 -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins direct dependencies to the versions in the working environment. It is not a complete lockfile for transitive dependencies. MFA runs in a **separate environment** and is not installed by this file.

External requirements:

1. **FFmpeg and ffprobe** on `PATH` for browser recordings and other audio formats.
2. **Montreal Forced Aligner** with the `mandarin_mfa` acoustic model. The extended dictionary is included at `mfa/mandarin_taiwan_catalog.dict`.
3. MFA and textbook data paths configured for the machine.

Check the external tools after setting `MFA_EXE`:

```powershell
ffmpeg -version
ffprobe -version
& $env:MFA_EXE version
```

Preloading requires the MFA environment's Python to provide the MFA, Kalpy, and OpenFst libraries used by `mfa_oneshot_worker.py`.

## Configuration

Set environment variables **before starting the application**. The application does not automatically load a `.env` file.

| Variable | Purpose / default |
| --- | --- |
| `MFA_EXE` | MFA executable; `D:\CodexData\mfa-runtime\env\Scripts\mfa.exe`. |
| `MFA_ROOT_DIR` | MFA model and state directory; `D:\CodexData\mfa-runtime\models`. |
| `MFA_PYTHON` | Python in the MFA environment for the preloaded worker; inferred from `MFA_EXE` if unset. |
| `TONE_MFA_WARM` | `1` enables preloading. Disabled by default when launching directly; enabled by `run_dev.ps1`. |
| `MANDARIN_ASSESSMENT_CONCURRENCY` | Maximum concurrent assessments/calibrations per API process; defaults to `2`. Must be a positive integer. Also sets the number of preloaded MFA worker slots. |
| `MANDARIN_SAMPLE_DATA_DIR` | Directory containing textbook MP3 files and `lessons-5-8_with_character_audio_synced.csv`. |
| `MANDARIN_WORD_LEVEL_PATH` | JSON word and phrase annotations indexed by sample ID. |
| `MANDARIN_PRACTICE_DATA_DIR` | Learner data directory; defaults to `practice_data/` in the project. |
| `MANDARIN_PRACTICE_DB` | SQLite path override; defaults to `<practice_data>/practice.sqlite3`. |
| `MANDARIN_RECORDINGS_DIR` | Recording directory override; defaults to `<practice_data>/recordings`. |
| `MANDARIN_USER_DATA_DIR` | Per-learner audio and log copies; defaults to `D:\hautran\Lab\mandarin-speaking data\Recorded Data\audio test\user data`. An empty string disables these copies. |
| `MANDARIN_ADMIN_PASSWORD` | Administrator password. If unset, administration is available only through loopback access. The HTTP Basic username is `admin`. |
| `MFA_BASE_DICT` | Source dictionary for `build_mfa_dictionary.py`; not required to use the included dictionary. |

Example for a different machine:

```powershell
$env:MFA_EXE = "C:\mfa\env\Scripts\mfa.exe"
$env:MFA_ROOT_DIR = "C:\mfa\models"
$env:MFA_PYTHON = "C:\mfa\env\python.exe"
$env:MANDARIN_SAMPLE_DATA_DIR = "D:\datasets\lessons"
$env:MANDARIN_WORD_LEVEL_PATH = "D:\datasets\lessons_5_8_word_level.json"
$env:MANDARIN_USER_DATA_DIR = "D:\mandarin-user-archive"
.\run_dev.ps1
```

Textbook sentence data defaults to the external directory `D:\hautran\Lab\mandarin-speaking data\Recorded Data\lessons_5_8_FINAL_content_verified_v3_finallllll`. Word annotations default to `Recorded Data\lessons_5_8_word_level.json` in its parent directory. Missing files produce warnings and leave textbook sentence practice without the corresponding data. The 53 Chapter 5 word and phrase recordings and five calibration sounds are included in this repository.

Assessment thresholds, tone rules, alignment handling, and difficulty levels are defined in `tone_v1_config.yaml`, the active runtime configuration.

## Concurrent requests

The server admits up to **two assessments or voice calibrations at once** by default. Different learners can be processed concurrently. Requests for the same learner run one at a time; deleting that learner waits for their active operation without stopping other learners.

Requests wait for an available slot for up to 30 seconds, then receive HTTP `429` with `Retry-After: 1`. Waiting requests do not have a guaranteed FIFO order, and this is not a persistent job queue.

To change the limit before starting the server:

```powershell
$env:MANDARIN_ASSESSMENT_CONCURRENCY = "2"
.\run_dev.ps1
```

With `TONE_MFA_WARM=1`, each slot has a separate one-shot alignment worker and dictionary working directory. Ready workers can align concurrently. Loading shared MFA models and the fallback `mfa align_one` command remain protected by the existing machine-wide file lock. Increasing the limit uses more CPU and memory; the previous estimate was about 340 MB per preloaded MFA process, in addition to application memory. Warm-up and fallback requests can still wait for shared MFA resources, and throughput does not scale linearly with this setting.

These admission and per-learner guards apply to one API process. Run one Uvicorn application worker with this configuration; multiple application processes would need shared coordination for learner operations. The HTTP request remains open until its result is saved. Archive file writes are briefly serialized to keep profiles and log headers consistent.

## Data and backups

`practice_data/` contains the SQLite database and original practice recordings. Pause data writes before backing up both the database and the audio directory. If `MANDARIN_USER_DATA_DIR` is enabled, back up that directory as well: it contains per-learner copies, assessment logs, and calibration logs. Calibration audio is discarded after analysis.

External datasets, the `env/` environment, external MFA models, and learner data are not tracked in Git. The `output/` and `outputs/` directories and tool caches are also ignored. Some evaluation tools use external datasets or alignment caches in `outputs/`; retain those caches for faster benchmark reruns.

## Project structure

| Path | Purpose |
| --- | --- |
| `main.py` | FastAPI endpoints and learner and administrator interfaces. |
| `assessment_concurrency.py` | Per-process admission limits and per-learner operation guards. |
| `audio_prep.py` | Audio decoding, quality checks, normalization, and silence trimming. |
| `capt_core.py` | Text processing, syllables, and expected tone contexts. |
| `tone_pipeline.py` | Calibration, MFA alignment, F0 extraction, and tone verification. |
| `tone_assessment.py` | Word and phrase feedback assembled from verification results. |
| `mfa_warm.py`, `mfa_oneshot_worker.py` | Preloaded alignment worker management. |
| `practice_store.py`, `user_archive.py` | Profiles, history, recordings, and archive copies. |
| `chapter5_word_practice/`, `assets/calibration/`, `mfa/` | Runtime data and resources. |
| `tests/`, `golden/` | Tests and regression reference data. |
| `evaluate_*.py`, `benchmark_tone_pipeline.py`, `compare_configs.py`, `word_regression.py` | Evaluation and algorithm regression tools. |
| `export_sentence_review.py`, `xlsx_reader.py`, `tools/` | Review exports, workbook reading, and calibration audio generation. |

## Validation

Install the additional test dependency:

```powershell
.\env\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run tests with separate temporary data directories. FFmpeg must be on `PATH`.

```powershell
$testData = Join-Path ([System.IO.Path]::GetTempPath()) ("mandarin-tests-" + [guid]::NewGuid())
$env:MANDARIN_PRACTICE_DATA_DIR = $testData
$env:MANDARIN_PRACTICE_DB = Join-Path $testData "practice.sqlite3"
$env:MANDARIN_RECORDINGS_DIR = Join-Path $testData "recordings"
$env:MANDARIN_USER_DATA_DIR = Join-Path $testData "archive"
$env:TONE_MFA_WARM = "0"
.\env\Scripts\python.exe -m unittest discover -s tests
```

Use a separate PowerShell window for testing so temporary data settings do not affect subsequent application runs.

When changing the algorithm or assessment thresholds, also compare against the existing regression snapshot:

```powershell
.\env\Scripts\python.exe word_regression.py --check
```

This command requires MFA and the external audio datasets referenced in `word_regression.py` and `golden/synthetic_manifest.csv`. Use `--freeze` only when intentionally accepting new results as the reference snapshot.

`compare_configs.py` accepts custom YAML files to compare against the active configuration; use `--help` to inspect the evaluation suites. Old trial configurations and draft audio have been removed from the working tree and remain available in Git history. [Historical tuning notes](docs/tone-tuning-history.md) explain earlier engineering decisions; they do not validate the current version.
