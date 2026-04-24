"""Centralized configuration loaded from environment variables."""

import os

from dotenv import load_dotenv
from questionary import Style

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
    """

    hf_token: str = os.getenv("HF_TOKEN", "")
    device: str = os.getenv("DEVICE", "cuda")
    compute_type: str = os.getenv("COMPUTE_TYPE", "float16")
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
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
    prompt_style: Style = Style([("pointer", "bold")])
