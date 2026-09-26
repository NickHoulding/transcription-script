"""Centralized configuration loaded from environment variables."""

import os
from typing import Literal, cast

from dotenv import load_dotenv
from questionary import Style
import logging
import warnings

load_dotenv()


class Config:
    """Static configuration loaded from environment variables at import time.

    Attributes:
        hf_token: Hugging Face API token for diarization model access.
        device: Inference device passed to WhisperX (e.g. ``"cuda"``, ``"cpu"``).
        compute_type: Model precision (e.g. ``"float16"``, ``"int8"``).
        batch_size: Number of audio chunks processed per transcription batch.
        default_model: WhisperX model used when the user makes no selection.
        transcription_models: Ordered list of available WhisperX model names.
        log_level: Level applied to noisy third-party loggers (e.g. ``"ERROR"``).
        warnings_enabled: Whether third-party warning filters are installed.
        warnings_action: Action passed to ``warnings.filterwarnings`` (e.g. ``"ignore"``).
        prompt_style: Questionary style applied to interactive CLI prompts.
    """

    # -------------------------------------------------------------------------
    # Top-level entry point
    # -------------------------------------------------------------------------

    @staticmethod
    def configure() -> None:
        """Apply logging and warning configuration. Call once at import time."""
        Config._configure_logging()
        Config._configure_warnings()

    # -------------------------------------------------------------------------
    # Hugging Face
    # -------------------------------------------------------------------------

    hf_token: str = os.getenv("HF_TOKEN", "")

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device: str = os.getenv("DEVICE", "cuda")
    compute_type: str = os.getenv("COMPUTE_TYPE", "float16")
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))

    # -------------------------------------------------------------------------
    # WhisperX model
    # -------------------------------------------------------------------------

    default_model: str = os.getenv("DEFAULT_MODEL", "medium.en")
    transcription_models: list[str] = [
        "tiny.en",
        "base.en",
        "small.en",
        "medium.en",
        "large-v2",
        "large-v3",
        "turbo",
    ]

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    log_level: str = os.getenv("LOG_LEVEL", "ERROR")

    _logging_modules: list[str] = [
        "whisperx",
        "whisperx.vads.pyannote",
        "whisperx.diarize",
        "pyannote",
        "lightning.pytorch",
        "lightning.pytorch.utilities.migration.utils",
    ]

    @staticmethod
    def _configure_logging() -> None:
        """Set the log level of noisy third-party loggers to ``Config.log_level``."""
        for module in Config._logging_modules:
            logging.getLogger(module).setLevel(Config.log_level)

    # -------------------------------------------------------------------------
    # Warnings
    # -------------------------------------------------------------------------

    warnings_enabled: bool = os.getenv("WARNINGS_ENABLED", "true").lower() == "true"
    warnings_action: str = os.getenv("WARNINGS_ACTION", "ignore")

    _warning_modules: list[str] = ["whisperx", "pyannote", "pyannote.audio.core.io"]

    @staticmethod
    def _configure_warnings() -> None:
        """Filter warnings from noisy third-party modules per ``Config.warnings_enabled``/``warnings_action``."""
        if not Config.warnings_enabled:
            return

        _action = cast(
            Literal["default", "error", "ignore", "always", "module", "once"],
            Config.warnings_action,
        )

        for module in Config._warning_modules:
            warnings.filterwarnings(_action, module=module)

    # -------------------------------------------------------------------------
    # UI style
    # -------------------------------------------------------------------------

    prompt_style: Style = Style([("pointer", "bold")])
