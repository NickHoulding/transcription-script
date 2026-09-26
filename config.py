"""Centralized configuration loaded from environment variables."""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv
from questionary import Style
import logging
import torch
import warnings

load_dotenv()

_PROJECT_ROOT: Path = Path(__file__).resolve().parent


class Config:
    """Static configuration loaded from environment variables at import time.

    Attributes:
        hf_token: Hugging Face API token for diarization model access.
        device: Inference device passed to WhisperX; auto-detected unless ``DEVICE`` is set.
        compute_type: Model precision; auto-detected by device unless ``COMPUTE_TYPE`` is set.
        batch_size: Number of audio chunks processed per transcription batch.
        transcription_models: Ordered list of available WhisperX model names.
        model_dir: The root-directory-constrained path where models are downloaded/cached.
        third_party_log_level: Level applied to suppress noisy third-party loggers.
        warnings_enabled: Whether third-party warning filters are installed.
        warnings_action: Action passed to ``warnings.filterwarnings`` (e.g. ``"ignore"``).
        log_dir: The root-directory-constrained path application log files are written to.
        log_level: Level applied to the application's own logger (e.g. ``"INFO"``).
        log_retention_count: Number of most-recent run log files kept in ``log_dir``.
        prompt_style: Questionary style applied to interactive CLI prompts.
    """

    # -------------------------------------------------------------------------
    # Top-level entry point
    # -------------------------------------------------------------------------

    @staticmethod
    def configure() -> None:
        """Apply logging and warning configuration. Call once at import time."""
        Config._configure_app_logging()
        Config._configure_third_party_logging()
        Config._configure_warnings()
        Config._configure_model_storage()

    # -------------------------------------------------------------------------
    # Hugging Face
    # -------------------------------------------------------------------------

    hf_token: str = os.getenv("HF_TOKEN", "")

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device: str = os.getenv("DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    compute_type: str = os.getenv("COMPUTE_TYPE") or (
        "float16" if device == "cuda" else "int8"
    )
    batch_size: int = int(os.getenv("BATCH_SIZE") or "16")

    # -------------------------------------------------------------------------
    # WhisperX model
    # -------------------------------------------------------------------------

    transcription_models: list[str] = [
        "tiny.en",
        "base.en",
        "small.en",
        "medium.en",
        "large-v2",
        "large-v3",
        "turbo",
    ]
    model_dir: Path = _PROJECT_ROOT / (os.getenv("MODEL_DIR") or "models")

    @staticmethod
    def _configure_model_storage() -> None:
        """Ensure ``Config.model_dir`` exists so models download/cache into the project."""
        Config.model_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Application logging
    # -------------------------------------------------------------------------

    log_dir: Path = _PROJECT_ROOT / (os.getenv("LOG_DIR") or "logs")
    log_level: str = os.getenv("LOG_LEVEL") or "INFO"
    log_retention_count: int = int(os.getenv("LOG_RETENTION_COUNT") or "10")

    @staticmethod
    def _configure_app_logging() -> None:
        """Write application logs for this run to a timestamped file under ``Config.log_dir``."""
        Config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file: Path = Config.log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log"

        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )

        root_logger = logging.getLogger()
        root_logger.setLevel(Config.log_level)
        root_logger.addHandler(handler)

        Config._prune_old_logs()

    @staticmethod
    def _prune_old_logs() -> None:
        """Delete old run log files beyond ``Config.log_retention_count``, keeping the most recent."""
        log_files = sorted(Config.log_dir.glob("*.log"), key=lambda p: p.name)
        excess = len(log_files) - Config.log_retention_count
        for old_file in log_files[: max(excess, 0)]:
            old_file.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Third-party logging
    # -------------------------------------------------------------------------

    third_party_log_level: str = os.getenv("THIRD_PARTY_LOG_LEVEL") or "ERROR"

    _third_party_logging_modules: list[str] = [
        "whisperx",
        "whisperx.vads.pyannote",
        "whisperx.diarize",
        "pyannote",
        "lightning.pytorch",
        "lightning.pytorch.utilities.migration.utils",
    ]

    @staticmethod
    def _configure_third_party_logging() -> None:
        """Set the log level of noisy third-party loggers to ``Config.third_party_log_level``.

        Mainly used to suppress noisy logs from project dependencies.
        """
        for module in Config._third_party_logging_modules:
            logging.getLogger(module).setLevel(Config.third_party_log_level)

    # -------------------------------------------------------------------------
    # Warnings
    # -------------------------------------------------------------------------

    warnings_enabled: bool = (os.getenv("WARNINGS_ENABLED") or "true").lower() == "true"
    warnings_action: str = os.getenv("WARNINGS_ACTION") or "ignore"

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
