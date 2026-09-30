"""Centralized configuration loaded from environment variables."""

import os
import socket
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _is_online(
    host: str = "huggingface.co", port: int = 443, timeout: float = 1.5
) -> bool:
    """Check for internet connectivity via a short-timeout TCP connect."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


_ONLINE: bool = _is_online()

if not _ONLINE:
    """huggingface_hub (used by faster-whisper and pyannote.audio) reads HF_HUB_OFFLINE once at import time, so this must be set before it's imported anywhere below. Otherwise every model load attempts a real network request first, hangs on DNS resolution/retries, and only falls back to the local cache after a long timeout."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datetime import datetime
from typing import Literal, cast

from faster_whisper.utils import available_models, download_model
from huggingface_hub.errors import LocalEntryNotFoundError
from questionary import Style
import logging
import torch
import warnings


def _is_model_cached(model: str, model_dir: Path) -> bool:
    """Check whether ``model`` is already downloaded into ``model_dir``."""
    try:
        download_model(model, local_files_only=True, cache_dir=str(model_dir))
        return True
    except LocalEntryNotFoundError:
        return False


def _resolve_transcription_models(model_dir: Path) -> tuple[list[str], set[str]]:
    """List available WhisperX models, and which of them are already cached.

    Returns:
        A tuple of (model names, cached model names). When offline, the model
        list is constrained to only cached models (so the second set then
        equals the first list).
    """
    available: list[str] = available_models()

    if not _ONLINE:
        cached_only = [
            model for model in available if _is_model_cached(model, model_dir)
        ]
        return cached_only, set(cached_only)

    cached = {model for model in available if _is_model_cached(model, model_dir)}
    return available, cached


class Config:
    """Static configuration loaded from environment variables at import time.

    Attributes:
        hf_token: Hugging Face API token for diarization model access.
        device: Inference device passed to WhisperX; auto-detected unless ``DEVICE`` is set.
        compute_type: Model precision; auto-detected by device unless ``COMPUTE_TYPE`` is set.
        batch_size: Number of audio chunks processed per transcription batch.
        model_dir: The root-directory-constrained path where models are downloaded/cached.
        transcription_models: Available WhisperX model names, resolved from faster-whisper. Constrained to already-cached models when offline.
        cached_transcription_models: Subset of ``transcription_models`` already downloaded.
        is_online: Whether internet connectivity was detected at import time.
        default_formats: Output formats pre-checked in the format selection prompt.
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

    model_dir: Path = _PROJECT_ROOT / (os.getenv("MODEL_DIR") or "models")
    transcription_models, cached_transcription_models = _resolve_transcription_models(
        model_dir
    )
    is_online: bool = _ONLINE

    @staticmethod
    def _configure_model_storage() -> None:
        """Ensure ``Config.model_dir`` exists so models download/cache into the project."""
        Config.model_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    default_formats: set[str] = {".txt", ".json"}

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
