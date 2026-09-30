"""Transcription script using WhisperX for ASR, alignment, and speaker diarization."""

import logging
import sys
from pathlib import Path

import questionary

from .config import Config
from .pipeline import TranscriptionPipeline
from .validate import (
    validate_hf_token,
    validate_num_speakers,
    validate_file_path,
    validate_save_path,
    validate_selected_formats,
    validate_transcription_models_available,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Gather inputs and hand off to the transcription pipeline."""
    try:
        validate_hf_token()
        validate_transcription_models_available()
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

        if not Config.is_online:
            logger.info("Network unavailable. Listing locally cached models.")
            print("[!] Network unavailable. Listing locally cached models.")

        selected_model: str = questionary.select(
            message="Select a model:",
            instruction="(move: arrow keys, submit: enter)",
            choices=[
                questionary.Choice(
                    title=(
                        f"{model} (cached)"
                        if Config.is_online
                        and model in Config.cached_transcription_models
                        else model
                    ),
                    value=model,
                )
                for model in Config.transcription_models
            ],
            qmark="❯",
            pointer="❯",
            style=Config.prompt_style,
        ).unsafe_ask()

        selected_formats: list[str] = questionary.checkbox(
            message="Select desired output format(s):",
            instruction="(move: arrow keys, toggle: space, submit: enter)",
            validate=validate_selected_formats,
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
