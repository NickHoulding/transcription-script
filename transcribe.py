"""Transcription script using WhisperX for ASR, alignment, and speaker diarization."""

import json
import logging
import sys
import warnings

# Suppress logging output before any whisperx/pyannote imports
logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("whisperx.vads.pyannote").setLevel(logging.ERROR)
logging.getLogger("whisperx.diarize").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("lighning.pytorch").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", module="whisperx")
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

import os
from typing import Any, cast

import numpy as np
import pandas as pd
import whisperx
from dotenv import load_dotenv
from whisperx.asr import FasterWhisperPipeline, TranscriptionResult
from whisperx.diarize import DiarizationPipeline

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

COMPUTE_TYPE: str = "float16"
DEVICE: str = "cuda"
BATCH_SIZE: int = 16
HF_TOKEN: str = os.getenv("HF_TOKEN") or ""
DEFAULT_TRANSCRIPTION_MODEL: str = "medium.en"
TRANSCRIPTION_MODELS: list[str] = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "large-v2",
    "large-v3",
    "turbo",
]


# =============================================================================
# Input helpers
# =============================================================================


def get_str_input(message: str = "") -> str:
    """Prompt the user for a string value and return the stripped result.

    Args:
      message: Optional prompt text displayed before the input cursor.

    Returns:
      The string entered by the user (whitespace-stripped).

    Raises:
      KeyboardInterrupt: Re-raised so callers can handle Ctrl-C cleanly.
    """
    try:
        input_val: str = input(f"{message}\n>>> ")
        print()
        return input_val.strip()
    except KeyboardInterrupt:
        print()
        raise


def get_int_input(message: str = "") -> int:
    """Prompt the user for an integer value and return it.

    Args:
      message: Optional prompt text displayed before the input cursor.

    Returns:
      The integer entered by the user.

    Raises:
      KeyboardInterrupt: Re-raised so callers can handle Ctrl-C cleanly.
      ValueError: If the entered value cannot be converted to an integer.
    """
    str_int: str = get_str_input(message=message)
    return int(str_int)


# =============================================================================
# Validation
# =============================================================================


def validate_file_path(file_path: str) -> bool:
    """Return True only if file_path points to an existing regular file.

    Args:
      file_path: The filesystem path to validate.

    Returns:
      True if the path is non-empty, exists, and is a regular file; False otherwise.
    """
    return (
        len(file_path) > 0 and os.path.exists(file_path) and os.path.isfile(file_path)
    )


def validate_hf_token() -> None:
    """Ensure HF_TOKEN is set; raise RuntimeError if it is empty.

    Raises:
      RuntimeError: If HF_TOKEN is not set or is an empty string.
    """
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to your .env file as HF_TOKEN=<your_token>."
        )


# =============================================================================
# Model selection
# =============================================================================


def select_transcription_model() -> str:
    """Interactively prompt the user to pick a WhisperX model by number.

    Displays the list of available models and reads an integer choice.
    Falls back to DEFAULT_TRANSCRIPTION_MODEL if the input is non-numeric
    or out of range.

    Returns:
      The name of the selected (or default) transcription model.

    Raises:
      KeyboardInterrupt: Re-raised so main() can handle Ctrl-C cleanly.
    """
    print("Available Transcription Models:")
    for index, model_name in enumerate(TRANSCRIPTION_MODELS):
        print(f"[{index + 1}] {model_name}")
    print()

    model: str = DEFAULT_TRANSCRIPTION_MODEL

    try:
        message: str = f"Select a model [1-{len(TRANSCRIPTION_MODELS)}]:"
        model_index: int = get_int_input(message=message)
        
        if model_index < 1 or model_index > len(TRANSCRIPTION_MODELS):
            raise IndexError(f"Choice {model_index} is out of range.")
        
        model = TRANSCRIPTION_MODELS[model_index - 1]
    except (ValueError, IndexError) as e:
        print(
            f"Invalid model choice ({e}). "
            f"Defaulting to transcription model '{DEFAULT_TRANSCRIPTION_MODEL}'."
        )
    except KeyboardInterrupt:
        raise

    return model


# =============================================================================
# Main pipeline
# =============================================================================


def main() -> None:
    """Run the full WhisperX transcription, alignment, and diarization pipeline.

    Prompts for the number of speakers, file path, and model, then runs each
    stage in sequence. All errors are caught and printed without a traceback;
    KeyboardInterrupt exits cleanly.
    """
    try:
        validate_hf_token()
    except RuntimeError as e:
        print(f"[ERROR] Configuration error: {e}")
        sys.exit(1)

    try:
        num_speakers_input: str = get_str_input(message="How many speakers are there?")

        if not num_speakers_input.isdigit():
            raise ValueError(
                f"Invalid number of speakers: '{num_speakers_input}'. "
                "Value must be a positive integer."
            )

        num_speakers: int = int(num_speakers_input)

        if num_speakers < 1:
            raise ValueError(
                f"Number of speakers must be at least 1, got {num_speakers}."
            )

        file_path_input: str = get_str_input(message="Enter file path:")

        if not validate_file_path(file_path_input):
            raise ValueError(
                f"Invalid file path: '{file_path_input}'. "
                "The path must point to an existing file."
            )

        file_path: str = file_path_input
        selected_model: str = select_transcription_model()

        print(f"[*] Loading transcription model '{selected_model}'...")
        try:
            transcription_model: FasterWhisperPipeline = whisperx.load_model(
                selected_model, device=DEVICE, compute_type=COMPUTE_TYPE
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load transcription model '{selected_model}': {e}"
            ) from e

        print("[OK] Transcription model successfully loaded.")

        try:
            audio: np.ndarray = whisperx.load_audio(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from '{file_path}': {e}") from e

        print("[OK] Audio successfully loaded.")

        print("[*] Transcribing...")
        try:
            transcription: TranscriptionResult = transcription_model.transcribe(
                audio, batch_size=BATCH_SIZE
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e

        print("[OK] Transcription complete.")

        print("[*] Loading alignment model...")
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=transcription["language"], device=DEVICE
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load alignment model: {e}") from e

        print("[OK] Alignment model loaded.")
        print("[*] Aligning audio segments...")
        try:
            aligned_transcription: dict[str, Any] = whisperx.align(
                transcription["segments"],
                align_model,
                metadata,
                audio,
                DEVICE,
            )
        except Exception as e:
            raise RuntimeError(f"Audio alignment failed: {e}") from e

        print("[OK] Audio alignment complete.")

        print("[*] Loading diarization model...")
        try:
            diarize_model: DiarizationPipeline = DiarizationPipeline(
                token=HF_TOKEN, device=DEVICE
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load diarization model: {e}") from e

        print("[OK] Diarization model loaded.")
        print("[*] Performing diarization...")
        try:
            segments: pd.DataFrame = cast(
                pd.DataFrame, diarize_model(audio=audio, num_speakers=num_speakers)
            )
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}") from e

        print("[OK] Diarization complete.")

        print("[*] Assigning speakers to segments...")
        try:
            result: dict[str, Any] = whisperx.assign_word_speakers(
                segments, aligned_transcription
            )
        except Exception as e:
            raise RuntimeError(f"Speaker assignment failed: {e}") from e

        print("[OK] Speaker segments successfully assigned.")

        print("\nTranscription:")
        for segment in result["segments"]:
            speaker: str = segment.get("speaker", "UNKNOWN")
            text: str = segment.get("text", "")
            print(f"{speaker}: {text}")

        print()
        print(json.dumps(result, indent=4))

    except KeyboardInterrupt:
        print("\n[CANCELLED] Interrupted by user.")
        sys.exit(0)
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
