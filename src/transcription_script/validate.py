"""Validation methods for the transcription script"""

import os
from .config import Config

# =============================================================================
# Runtime validation
# =============================================================================


def validate_hf_token() -> None:
    """Ensure HF_TOKEN is set; raise RuntimeError if it is empty.

    Raises:
        RuntimeError: If HF_TOKEN is not set or is an empty string.
    """
    if not Config.hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to your .env file as HF_TOKEN=<your_token>."
        )


def validate_transcription_models_available() -> None:
    """Ensure at least one transcription model is selectable; raise RuntimeError otherwise.

    Raises:
        RuntimeError: If no internet connection was detected and no models are cached locally.
    """
    if not Config.transcription_models:
        raise RuntimeError(
            "No transcription models are available offline and no internet connection "
            f"was detected. Connect to the internet, or download a model into "
            f"'{Config.model_dir}' first."
        )


# =============================================================================
# Input validation
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


def validate_selected_formats(formats: list[str]) -> bool | str:
    if not formats:
        return "At least one output format must be selected."
    return True
