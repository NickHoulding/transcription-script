"""Transcription script using WhisperX for ASR, alignment, and speaker diarization."""

import os
import sys

import questionary

from config import Config
from pipeline import TranscriptionPipeline

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
        input_val: str = questionary.text(message=message).ask()
        print()
        return input_val.strip().strip("\"'")
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


def validate_save_path(save_path: str) -> bool:
    """Return True only if save_path points to an existing directory.

    Args:
        save_path: The filesystem path to validate.

    Returns:
        True if the path is non-empty, exists, and is a directory; False otherwise.
    """
    return len(save_path) > 0 and os.path.exists(save_path) and os.path.isdir(save_path)


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
    """Interactively prompt the user to pick a WhisperX model by number.

    Displays the list of available models and reads an integer choice.
    Falls back to DEFAULT_TRANSCRIPTION_MODEL if the input is non-numeric
    or out of range.

    Returns:
        The name of the selected (or default) transcription model.

    Raises:
        KeyboardInterrupt: Re-raised so main() can handle Ctrl-C cleanly.
    """
    model: str = Config.default_model

    try:
        model = questionary.select(
            message="Select a model", choices=Config.transcription_models
        ).ask()
    except KeyboardInterrupt:
        raise
    except (ValueError, IndexError) as e:
        print(
            f"Invalid model choice ({e}). "
            f"Defaulting to transcription model '{Config.default_model}'."
        )
    print()

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

        file_path_input: str = get_str_input(
            message="Enter path to input file (absolute):"
        )

        if not validate_file_path(file_path_input):
            raise ValueError(
                f"Invalid file path: '{file_path_input}'. "
                "The path must point to an existing file."
            )

        save_path_input: str = get_str_input(
            message="Enter path to existing save directory (absolute):"
        )

        if not validate_save_path(save_path_input):
            raise ValueError(
                f"Invalid save path: '{save_path_input}'. "
                "The path must point to an existing directory."
            )

        selected_model: str = select_transcription_model()

    except KeyboardInterrupt:
        print("\n[CANCELLED] Interrupted by user.")
        sys.exit(0)
    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}")
        sys.exit(1)

    TranscriptionPipeline(
        file_path=file_path_input,
        save_path=save_path_input,
        num_speakers=num_speakers,
        model=selected_model,
    ).run()


if __name__ == "__main__":
    main()
