# Transcription Script

![Python Version](https://img.shields.io/badge/python-3.11.9-blue.svg)
![WhisperX](https://img.shields.io/badge/WhisperX-3.8+-green.svg)
![Device](https://img.shields.io/badge/device-CPU%20%7C%20CUDA%20(auto--detected)-orange.svg)

> Automated audio/video transcription with speaker diarization, powered by WhisperX and pyannote.audio

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [About the Script](#about-the-script)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Format](#output-format)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

---

## Problem Statement

I built this script to help my father with a personal project: writing an autobiography about his life. He's been conducting extensive interviews with family members, friends, and colleagues to gather stories and perspectives for his book. Each conversation captures invaluable memories and insights, but they also generate hours upon hours of audio recordings that need to be transcribed.

**The problem we faced:**

When he first started this project, we quickly realized that manual transcription was going to be a bottleneck. Professional transcription services would cost thousands of dollars for the volume of recordings he had, and doing it manually would take hours neither of us had. Beyond the raw transcription work, we also needed to track who was saying what throughout these conversations. My father's interviews often include multiple people reminiscing together, and distinguishing between speakers manually meant constant rewinding and note-taking.

We needed a solution that could:
- Process hours of interview recordings quickly and accurately
- Automatically identify and label different speakers throughout conversations
- Generate clean, readable transcripts that he could reference while writing
- Preserve the structure and flow of multi-party discussions

**My solution:**

I developed this automated transcription pipeline that combines state-of-the-art speech recognition (WhisperX) with speaker diarization (pyannote.audio). The script processes audio files in minutes instead of hours, automatically separates and labels different speakers, and outputs both human-readable transcripts and structured JSON data for programmatic analysis. What would have taken weeks of manual work now happens in an afternoon.

**Broader applications:**

While I built this specifically for my father's autobiography project, the script is useful well beyond memoir research. It works equally well for transcribing lecture recordings, business meetings, podcast episodes, focus groups, oral history projects, or any audio/video content where knowing "who said what" matters. The core challenge: converting multi-speaker audio into searchable, structured text, appears across countless domains.

---

## About the Script

This script is a Python-based command-line tool that converts audio and video files into speaker-labeled transcripts. It leverages WhisperX, an implementation of OpenAI's Whisper model, combined with pyannote.audio's speaker diarization capabilities to deliver accurate transcriptions with automatic speaker identification.

The script handles the complete pipeline:
- **Transcription** - Converts speech to text using state-of-the-art Whisper models
- **Alignment** - Refines word-level timestamps for precise timing
- **Diarization** - Identifies and separates different speakers throughout the audio
- **Speaker Assignment** - Matches transcribed text segments to identified speakers
- **Output Generation** - Creates both human-readable (.txt) and structured (.json) outputs

Unlike simple transcription tools, this script preserves speaker context, making it ideal for interviews, conversations, and multi-party discussions where knowing "who said what" is essential.

---

## Tech Stack

### **Core Technologies**
- **Python 3.11.9** - Modern Python with type hints and improved performance
- **WhisperX 3.8+** - Enhanced Whisper implementation with alignment and diarization
- **PyTorch** - Deep learning framework for model inference; also used to auto-detect CUDA availability
- **CUDA** - Optional GPU acceleration; auto-detected at startup, falls back to CPU when unavailable

### **Audio & Speech Processing**
- **faster-whisper** - Optimized Whisper implementation using CTranslate2
- **pyannote.audio** - Speaker diarization and voice activity detection
- **ffmpeg** - Audio/video format handling and preprocessing

### **Machine Learning Models**
- **OpenAI Whisper** - Automatic speech recognition (multiple model sizes available)
  - Options: `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v2`, `large-v3`, `turbo`
- **pyannote/speaker-diarization-community-1** - Speaker diarization pipeline (segmentation, embedding, and clustering bundled together)

### **Utilities**
- **python-dotenv** - Environment variable management
- **questionary** - Interactive CLI prompts
- **rich** - Spinner/progress output during pipeline stages
- **NumPy** - Numerical operations for audio processing
- **pandas** - Diarization result handling

---

## Key Features

### **Flexible Model Selection**
- **7 Whisper model options** - Choose speed vs. accuracy tradeoff
  - `tiny.en` - Fastest, lowest accuracy
  - `medium.en` - Balanced performance (default)
  - `turbo` - Latest model with improved speed
- Interactive model selection at runtime

### **Self-Contained Model Storage**
- **Project-local model cache** - Whisper, alignment, and diarization models download into a top-level `models/` directory instead of your global Hugging Face/torch caches, so the project's model footprint stays self-contained and easy to locate or wipe
- **Standard cache reuse** - Each library's own caching (Hugging Face Hub blobs/etags, `torch.hub`) is preserved, so models are only downloaded once

### **Speaker Diarization**
- **Automatic speaker identification** - Distinguishes between different voices
- **Configurable speaker count** - Specify expected number of speakers for better accuracy
- **Word-level speaker assignment** - Precise attribution of individual words to speakers
- **Handles overlapping speech** - Separates simultaneous speakers

### **Robust Pipeline**
- **Multi-stage processing** - Transcription → alignment → diarization → speaker assignment
- **Word-level timestamp alignment** - Precise timing information for each word
- **Language detection** - Automatically detects spoken language for alignment
- **Error handling** - Graceful failure with informative error messages, both printed to console and logged with a full traceback

### **Dual Output Formats**
- **Human-readable TXT** - Speaker-labeled transcript for easy reading
- **Structured JSON** - Complete data with timestamps, confidence scores, and metadata
- **Consistent naming** - Output files named after input file stem

### **Interactive CLI**
- **Guided prompts** - Step-by-step input collection with validation
- **Path validation** - Ensures input files and output directories exist; accepts relative and absolute paths
- **Progress indicators** - Real-time feedback on pipeline stages
- **Keyboard interrupt handling** - Clean exit with Ctrl+C

### **Structured Logging**
- **One log file per run** - Written to a top-level `logs/` directory, timestamped by when the run started
- **Automatic retention** - Old run logs beyond a configurable count are pruned automatically
- **Console stays clean** - Logging runs alongside the existing console/spinner output rather than replacing it
- **Independent log levels** - The application's own log level and the level used to suppress noisy third-party loggers are configured separately

### **Centralized, Blank-Safe Configuration**
- **Single source of truth** - All tunables live as static fields on `Config` in `config.py`, loaded from environment variables at import time
- **Every variable is optional** - Every setting in `.env` can be left blank to fall back to its built-in default; The HuggingFace token is the only required field
- **Auto-detected hardware settings** - `DEVICE` and `COMPUTE_TYPE` auto-detect sensible values (CUDA/float16 if a GPU is available, otherwise CPU/int8) unless explicitly overridden

### **Hardware Acceleration**
- **Auto-detected CUDA support** - Uses an NVIDIA GPU automatically when available for a large speedup over CPU
- **CPU fallback** - Runs fully on CPU (with an int8 compute type) when no GPU is present, no configuration needed
- **Configurable compute type** - Override the auto-detected precision (e.g. `float16`, `int8`, `float32`) via `COMPUTE_TYPE`
- **Batch processing** - Efficient inference with a configurable batch size

---

## Prerequisites

### **Hardware Requirements**

- **CUDA-capable NVIDIA GPU** (optional, recommended) - `DEVICE` auto-detects and uses CUDA when available for a large speedup; the script runs fine on CPU with no configuration if no GPU is present

### **Software Requirements**

- **Python 3.11.9** (exact version required due to dependencies)
- **CUDA Toolkit 11.8+ or 12.x** (only needed if using a GPU)
  - Download: https://developer.nvidia.com/cuda-downloads
  - Verify: `nvcc --version` or `nvidia-smi`
- **ffmpeg** (for audio/video processing)
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: Download from https://ffmpeg.org/download.html

### **HuggingFace Account**

- **HuggingFace token** (required for speaker diarization)
  - Model used: `pyannote/speaker-diarization-community-1`
  - Create account: https://huggingface.co/join
  - Generate token: https://huggingface.co/settings/tokens (read access sufficient)
  - Accept the model agreement: https://huggingface.co/pyannote/speaker-diarization-community-1

---

## Getting Started

### **Installation**

#### **Method 1: Using `uv` (Recommended)**

`uv` is a fast Python package manager that simplifies dependency management.

1. **Install `uv`**
   ```bash
   # Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd transcription-script
   ```

3. **Install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

4. **Verify installation**
   ```bash
   uv run python -c "import transcription_script"
   ```
   No output means everything imported successfully. This is a purely interactive prompt-driven script (see [Usage](#usage)).

#### **Method 2: Using `pip` (Alternative)**

If you encounter issues with `uv` on Windows or prefer traditional pip:

1. **Ensure Python 3.11.9 is installed**
   ```bash
   python --version  # Should show 3.11.9
   ```

2. **Clone or download the project**
   ```bash
   cd transcription-script
   ```

3. **Install the project (editable), globally or in a venv**
   ```bash
   # Option B: Virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .

   # Option A: Global installation (not recommended, only if other options aren't working)
   pip install -e .
   ```
   `-e .` installs this project itself (not just its dependencies), which is what registers the `transcribe` console script and makes the `transcription_script` package importable.

4. **Run the script**
   ```bash
   transcribe  # Or: python -m transcription_script
   ```

### **Environment Setup**

1. **Copy the example environment file**
   ```bash
   cp .env.example .env
   ```

2. **Add your HuggingFace token**

   Open `.env` and set:
   ```env
   HF_TOKEN=YOUR_HF_TOKEN_HERE
   ```
   Replace `YOUR_HF_TOKEN_HERE` with your actual token from https://huggingface.co/settings/tokens. This is the only required variable. Every other variable in `.env.example` is optional and can be left blank to use its built-in default (see [Configuration](#configuration)).

3. **Verify token is loaded**
   ```bash
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Token found!' if os.getenv('HF_TOKEN') else 'Token missing!')"
   ```

### **First Run**

On first execution, the script downloads the selected transcription model, the language-specific alignment model, and the diarization pipeline into a project-local `models/` directory. This is a one-time download per model. Subsequent runs reuse the cached files. See [Self-Contained Model Storage](#key-features) and the `MODEL_DIR` variable in [Configuration](#configuration) to change where models are cached.

---

## Configuration

All configuration lives in `config.py` as static fields on the `Config` class, loaded from environment variables at import time. Copy `.env.example` to `.env` and fill in `HF_TOKEN`. Every other variable is optional and falls back to the default below when left blank.

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | **Yes** | - | Hugging Face API token, needed for diarization model access |
| `DEVICE` | No | auto-detected (`cuda` if available, else `cpu`) | Inference device passed to WhisperX |
| `COMPUTE_TYPE` | No | auto-detected (`float16` on `cuda`, `int8` on `cpu`) | Model precision passed to WhisperX |
| `BATCH_SIZE` | No | `16` | Number of audio chunks processed per transcription batch |
| `MODEL_DIR` | No | `models` (project root) | Top-level directory models are downloaded/cached into |
| `THIRD_PARTY_LOG_LEVEL` | No | `ERROR` | Log level applied to noisy third-party loggers (WhisperX, pyannote, PyTorch Lightning), mainly used to suppress noise. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `WARNINGS_ENABLED` | No | `true` | Whether third-party warning filters are installed. Choices: `true`, `false` |
| `WARNINGS_ACTION` | No | `ignore` | Action applied to filtered warnings. Choices: `default`, `error`, `ignore`, `always`, `module`, `once` |
| `LOG_DIR` | No | `logs` (project root) | Top-level directory application log files are written to |
| `LOG_LEVEL` | No | `INFO` | Log level for the application's own logger. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_RETENTION_COUNT` | No | `10` | Number of most-recent run log files kept in `LOG_DIR`; older ones are pruned automatically |

`MODEL_DIR` and `LOG_DIR` are always resolved relative to the project root, not the current working directory, so they land in the same place regardless of where you run the script from.

---

## Usage

### **Running the Script**

Start the interactive transcription pipeline:

```bash
# If using uv:
uv run transcribe
# Or: uv run python -m transcription_script

# If using pip/global install:
transcribe
# Or: python -m transcription_script
```

### **Interactive Prompts**

The script will guide you through 4 steps:

#### **1. Specify Number of Speakers**
```
❯ Enter number of speakers: 2
```
- Enter the number of distinct speakers in the audio
- Must be a positive integer (1 or greater)
- More accurate diarization with correct speaker count

#### **2. Provide Input File Path**
```
❯ Enter path to input file: /home/user/recordings/interview_2024.mp3
```
- Provide the path (relative or absolute) to your audio or video file
- Supported formats: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.avi`, `.mov`, etc.
- Tab completion is available for filesystem paths
- File must exist or an inline error will prompt you to re-enter

#### **3. Specify Output Directory**
```
❯ Enter path to existing save directory: /home/user/transcripts/
```
- Provide the path (relative or absolute) to an existing directory
- Output files will be saved here with `_transcription.txt` and `_transcription.json` suffixes
- Directory must exist before running

#### **4. Select Transcription Model**
```
❯ Select a model: (Use arrow keys)
❯ tiny.en
  base.en
  small.en
  medium.en
  large-v2
  large-v3
  turbo
```
- Navigate with arrow keys and confirm with Enter
- Smaller models = faster, less accurate
- Larger models = slower, more accurate

### **Pipeline Stages**

Once inputs are provided, the script executes these stages:

1. **Model Loading** - Loads selected Whisper model
2. **Audio Loading** - Reads and preprocesses input file
3. **Transcription** - Converts speech to text
4. **Alignment Model Loading** - Loads language-specific alignment model
5. **Audio Alignment** - Refines word-level timestamps
6. **Diarization Model Loading** - Loads pyannote speaker diarization models
7. **Diarization** - Identifies speaker segments
8. **Speaker Assignment** - Matches text to speakers
9. **File Writing** - Generates `.txt` and `.json` outputs

Each stage prints its label and elapsed time on completion:
```
Loading transcription model 'medium.en' (4.3s)
Loading audio (0.8s)
Transcribing (2m 15s)
...

Total elapsed time (5m 23s)
```

Each of these lines is also written to that run's log file under `logs/` (see [Configuration](#configuration)), alongside the run's parameters and, on failure, a full traceback.

### **Example Session**

```bash
$ uv run transcribe

❯ Enter number of speakers: 2
❯ Enter path to input file: /home/nick/media/interview_john_doe.mp3
❯ Enter path to existing save directory: /home/nick/transcripts/
❯ Select a model: medium.en
Loading transcription model 'medium.en' (4.3s)
Loading audio (0.8s)
Transcribing (2m 15s)
Loading alignment model (1.2s)
Aligning audio segments (45.3s)
Loading diarization model (3.1s)
Performing diarization (1m 32s)
Assigning speakers to segments (0.4s)
Writing TXT file (0.1s)
Writing JSON file (0.2s)
Total elapsed time (5m 23s)
```

---

## Output Format

### **TXT Output (`filename_transcription.txt`)**

Human-readable format with speaker labels:

```
SPEAKER_00: Hello, thanks for joining me today. Can you tell me about your early years?
SPEAKER_01: Of course. I was born in 1952 in a small town in Ohio. My parents ran a general store.
SPEAKER_00: What was that like growing up in that environment?
SPEAKER_01: It was formative. I learned the value of hard work from watching them serve the community every day.
SPEAKER_00: That's a wonderful memory. Did you have siblings?
SPEAKER_01: Yes, I had two brothers and a sister. We all helped out in the store after school.
```

**Format details:**
- Each line represents a continuous speech segment by one speaker
- Speaker labels: `SPEAKER_00`, `SPEAKER_01`, `SPEAKER_02`, etc.
- Labels are assigned automatically and consistently throughout the transcript
- No timestamps (for clean reading experience)

### **JSON Output (`filename_transcription.json`)**

Structured format with complete metadata:

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 4.8,
      "text": " Hello, thanks for joining me today. Can you tell me about your early years?",
      "words": [
        {
          "word": "Hello",
          "start": 0.5,
          "end": 0.9,
          "score": 0.89,
          "speaker": "SPEAKER_00"
        },
        {
          "word": "thanks",
          "start": 1.0,
          "end": 1.3,
          "score": 0.92,
          "speaker": "SPEAKER_00"
        }
      ],
      "speaker": "SPEAKER_00"
    },
    {
      "start": 5.2,
      "end": 12.4,
      "text": " Of course. I was born in 1952 in a small town in Ohio. My parents ran a general store.",
      "words": [...],
      "speaker": "SPEAKER_01"
    }
  ],
  "word_segments": [...]
}
```

**JSON structure:**
- `segments[]` - Array of speech segments with speaker labels
  - `start` - Segment start time (seconds)
  - `end` - Segment end time (seconds)
  - `text` - Transcribed text for the segment
  - `speaker` - Speaker label (e.g., `SPEAKER_00`)
  - `words[]` - Word-level details with timestamps and confidence scores
    - `word` - Individual word
    - `start` / `end` - Word-level timestamps
    - `score` - Transcription confidence (0.0 - 1.0)
    - `speaker` - Speaker attribution for the word
- `word_segments[]` - Alternative word-level representation

**Use cases:**
- TXT: Reading, editing, sharing with non-technical users
- JSON: Data analysis, timestamp navigation, confidence filtering, programmatic processing

---

## Architecture

### **Pipeline Overview**

The script implements a four-stage pipeline combining multiple state-of-the-art models:

```
Audio/Video Input
      ↓
[1] Transcription (Whisper)
      ↓
[2] Alignment (whisperx align)
      ↓
[3] Diarization (pyannote.audio)
      ↓
[4] Speaker Assignment (whisperx)
      ↓
TXT + JSON Output
```

### **Stage Breakdown**

**1. Transcription (ASR)**
- Uses faster-whisper (CTranslate2-optimized Whisper)
- Converts audio to text with approximate timestamps
- Language detection for multilingual support
- Configurable model size for speed/accuracy tradeoff

**2. Alignment**
- Refines Whisper's timestamps using forced alignment
- Loads language-specific phoneme models
- Produces word-level timestamps with higher accuracy
- Essential for precise speaker attribution

**3. Diarization**
- Pyannote.audio identifies "who spoke when"
- Voice activity detection (VAD) finds speech regions
- Speaker embedding extraction for each segment
- Clustering to group segments by speaker
- Outputs speaker timeline independent of transcription

**4. Speaker Assignment**
- Matches diarization timeline with aligned transcription
- Assigns speaker labels to each word and segment
- Handles overlapping speech and speaker transitions
- Produces final speaker-labeled transcript

### **Why WhisperX?**

WhisperX integrates three separate models (Whisper, alignment, diarization) into a cohesive pipeline with:
- **Better timestamps** - Word-level accuracy via alignment
- **Speaker labels** - Native diarization integration
- **Production-ready** - Optimized inference with CTranslate2
- **Easy-to-use** - Single library for the full pipeline

### **Error Handling**

The script employs defensive programming:
- **Pre-flight validation** - Checks file paths, HF token, speaker count before processing
- **Stage-by-stage error catching** - Each pipeline stage wrapped in try/except
- **Informative messages** - Errors indicate which stage failed and why, printed to console and logged with a full traceback to that run's log file
- **Clean exits** - KeyboardInterrupt (Ctrl+C) exits gracefully without traceback
- **No partial outputs** - Files only written if full pipeline succeeds

---

## Troubleshooting

### **CUDA / GPU Issues**

**Problem:** `RuntimeError: CUDA out of memory`
- **Cause:** Selected model requires more VRAM than available
- **Solution:** Choose a smaller model (`tiny.en`, `base.en`, `small.en`)
- **Solution:** Reduce `BATCH_SIZE` in `.env` (default: 16)
- **Solution:** Close other GPU-using applications

**Problem:** Script runs on CPU when you expected GPU (slow)
- **Cause:** `torch.cuda.is_available()` returns `False`. PyTorch isn't built with CUDA support, or CUDA drivers are missing. `DEVICE` auto-detects this and falls back to `cpu`/`int8` automatically, so the script still runs, just slower.
- **Solution:** Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **Solution:** Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
- **Solution:** Verify CUDA installation: `nvidia-smi`
- **Solution:** Once fixed, `DEVICE`/`COMPUTE_TYPE` will auto-detect CUDA again on the next run. No `.env` change needed unless you want to force one or the other explicitly

### **HuggingFace Token Issues**

**Problem:** `RuntimeError: HF_TOKEN is not set`
- **Cause:** `.env` file missing or token not set
- **Solution:** Copy `.env.example` to `.env` and set `HF_TOKEN=YOUR_HF_TOKEN_HERE`
- **Solution:** Verify token: `cat .env` (Linux/Mac) or `type .env` (Windows)

**Problem:** `401 Client Error: Unauthorized for url: https://huggingface.co/...`
- **Cause:** Invalid token or model access not granted
- **Solution:** Regenerate token at https://huggingface.co/settings/tokens
- **Solution:** Accept the model agreement: https://huggingface.co/pyannote/speaker-diarization-community-1

### **Windows-Specific Issues**

**Problem:** `ffmpeg` not found
- **Cause:** ffmpeg not in system PATH
- **Solution:** Download ffmpeg from https://ffmpeg.org/download.html
- **Solution:** Add ffmpeg `bin` folder to PATH environment variable
- **Solution:** Verify: `ffmpeg -version` in Command Prompt

### **Model Download Issues**

**Problem:** Models download slowly or fail
- **Cause:** Large model files downloading over slow connection
- **Solution:** Be patient on first run (one-time download)
- **Solution:** Check disk space: models cache to the project-local `models/` directory (or wherever `MODEL_DIR` points)
- **Solution:** Manual download: Visit https://huggingface.co/models and download into the `models/` directory

### **Audio Processing Issues**

**Problem:** "Failed to load audio" error
- **Cause:** Unsupported audio format or corrupted file
- **Solution:** Convert to common format: `ffmpeg -i input.xyz output.mp3`
- **Solution:** Verify file plays in media player before processing
- **Solution:** Check file permissions (must be readable)

**Problem:** Transcription is inaccurate
- **Cause:** Poor audio quality, background noise, or wrong model
- **Solution:** Use larger model (`large-v3` for best accuracy)
- **Solution:** Pre-process audio: normalize volume, reduce noise
- **Solution:** Verify correct language (script assumes English models by default)

**Problem:** Speaker labels are wrong (speakers confused)
- **Cause:** Similar voices, overlapping speech, or incorrect speaker count
- **Solution:** Specify accurate speaker count (critical for diarization)
- **Solution:** Ensure speakers have distinct voices and minimal overlap
- **Solution:** Pre-process audio: separate channels if stereo recording with 1 speaker per channel

### **Memory Issues**

**Problem:** System runs out of RAM (not VRAM)
- **Cause:** Long audio files create large intermediate data structures
- **Solution:** Split long files into chunks (e.g., 30-60 minute segments)
- **Solution:** Close other memory-intensive applications
- **Solution:** Use smaller model (less memory overhead)

---

<div align="center">
  <strong>Transcription Script</strong>
  <br>
</div>
