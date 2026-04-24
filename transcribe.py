"""Transcription script using WhisperX for ASR, alignment, and speaker diarization."""

import os
import sys

import questionary

from config import Config
from pipeline import TranscriptionPipeline

# =============================================================================
# Validation
# =============================================================================


def validate_num_speakers(val: str) -> bool | str:
    stripped = val.strip()
    if not stripped.isdigit():
        return "Please enter a positive integer."
    if int(stripped) < 1:
        return "Must be at least 1."
    return True


def validate_file_path(file_path: str) -> bool | str:
    if not file_path:
        return "Path cannot be empty."
    if not os.path.exists(file_path):
        return "File does not exist."
    if not os.path.isfile(file_path):
        return "Path must point to a file, not a directory."
    return True


def validate_save_path(save_path: str) -> bool | str:
    if not save_path:
        return "Path cannot be empty."
    if not os.path.exists(save_path):
        return "Directory does not exist."
    if not os.path.isdir(save_path):
        return "Path must point to a directory, not a file."
    return True


def validate_hf_token() -> None:
    """Ensure HF_TOKEN is set; raise RuntimeError if it is empty.

    Raises:
        RuntimeError: If HF_TOKEN is not set or is an empty string.
    """
    if not Config.hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to your .env file as HF_TOKEN=<your_token>."
        )


# =============================================================================
# Model selection
# =============================================================================


def select_transcription_model() -> str:
    """Interactively prompt the user to pick a WhisperX model.

    Falls back to DEFAULT_TRANSCRIPTION_MODEL if selection fails.

    Returns:
        The name of the selected (or default) transcription model.

    Raises:
        KeyboardInterrupt: Propagates so main() can handle Ctrl-C cleanly.
    """
    model: str = Config.default_model

    try:
        model = questionary.select(
            message="Select a model:",
            choices=Config.transcription_models,
            qmark="❯",
            pointer="❯",
            style=Config.prompt_style,
        ).unsafe_ask()
    except (ValueError, IndexError) as e:
        print(
            f"Invalid model choice ({e}). "
            f"Defaulting to transcription model '{Config.default_model}'."
        )

    return model


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Gather inputs and hand off to the transcription pipeline."""
    try:
        validate_hf_token()
    except RuntimeError as e:
        print(f"[ERROR] Configuration error: {e}")
        sys.exit(1)

    try:
        num_speakers_input: str = questionary.text(
            message="Enter number of speakers:",
            validate=validate_num_speakers,
            qmark="❯",
            style=Config.prompt_style,
        ).unsafe_ask()
        num_speakers: int = int(num_speakers_input.strip())

        file_path_input: str = questionary.path(
            message="Enter path to input file (absolute):",
            validate=validate_file_path,
            qmark="❯",
            style=Config.prompt_style,
        ).unsafe_ask()

        save_path_input: str = questionary.path(
            message="Enter path to existing save directory (absolute):",
            validate=validate_save_path,
            qmark="❯",
            style=Config.prompt_style,
        ).unsafe_ask()

        selected_model: str = select_transcription_model()

    except KeyboardInterrupt:
        print("\n[CANCELLED] Interrupted by user.")
        sys.exit(0)

    TranscriptionPipeline(
        file_path=file_path_input,
        save_path=save_path_input,
        num_speakers=num_speakers,
        model=selected_model,
    ).run()


if __name__ == "__main__":
    main()
