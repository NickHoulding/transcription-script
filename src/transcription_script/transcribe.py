"""Transcription script using WhisperX for ASR, alignment, and speaker diarization."""

import logging
import os
import sys
from pathlib import Path

import questionary

from .config import Config
from .pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)

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
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        return "File does not exist."
    if not os.path.isfile(file_path):
        return "Path must point to a file, not a directory."
    return True


def validate_save_path(save_path: str) -> bool | str:
    if not save_path:
        return "Path cannot be empty."
    save_path = os.path.expanduser(save_path)
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
# Main
# =============================================================================


def main() -> None:
    """Gather inputs and hand off to the transcription pipeline."""
    try:
        validate_hf_token()
    except RuntimeError as e:
        logger.error("Configuration error: %s", e, exc_info=True)
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

        file_path_input: str = str(
            Path(
                questionary.path(
                    message="Enter path to input file:",
                    validate=validate_file_path,
                    qmark="❯",
                    style=Config.prompt_style,
                ).unsafe_ask()
            )
            .expanduser()
            .resolve()
        )

        save_path_input: str = str(
            Path(
                questionary.path(
                    message="Enter path to existing save directory:",
                    validate=validate_save_path,
                    qmark="❯",
                    style=Config.prompt_style,
                ).unsafe_ask()
            )
            .expanduser()
            .resolve()
        )

        selected_model: str = questionary.select(
            message="Select a model:",
            choices=Config.transcription_models,
            qmark="❯",
            pointer="❯",
            style=Config.prompt_style,
        ).unsafe_ask()

        selected_formats: list[str] = questionary.checkbox(
            message="Select desired output format(s)",
            choices=[
                questionary.Choice(fmt, checked=fmt in Config.default_formats)
                for fmt in TranscriptionPipeline.available_formats()
            ],
            qmark="❯",
            pointer="❯",
            style=Config.prompt_style,
        ).unsafe_ask()

    except KeyboardInterrupt:
        logger.info("Run cancelled by user.")
        print("\n[CANCELLED] Interrupted by user.")
        sys.exit(0)

    TranscriptionPipeline(
        file_path=file_path_input,
        save_path=save_path_input,
        num_speakers=num_speakers,
        model=selected_model,
        formats=selected_formats,
    ).run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)
