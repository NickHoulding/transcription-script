""""""

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

import os
from typing import Any, cast

import numpy as np
import pandas as pd
import whisperx
from whisperx.asr import FasterWhisperPipeline, TranscriptionResult
from whisperx.diarize import DiarizationPipeline

COMPUTE_TYPE: str = "float16"
DEVICE: str = "cuda"
BATCH_SIZE: int = 16
HF_TOKEN: str = ""
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


def get_str_input(message: str = "") -> str:
    """"""
    input_val = input(f"{message}:\n>>> ")
    print()
    return input_val


def get_int_input(message: str = "") -> int:
    """"""
    input_val = input(f"{message}:\n>>> ")
    print()
    return int(input_val)


def validate_file_path(file_path: str) -> bool:
    """"""
    return (
        len(file_path) > 0 and os.path.exists(file_path) and os.path.isfile(file_path)
    )


def select_transcription_model() -> str:
    """"""
    print("Available Transcription Models:")

    for index, model_name in enumerate(TRANSCRIPTION_MODELS):
        print(f"[{index + 1}] {model_name}")

    print()
    model: str = DEFAULT_TRANSCRIPTION_MODEL

    try:
        model_index: int = get_int_input(message="Select a model")
        model = TRANSCRIPTION_MODELS[model_index - 1]
    except (ValueError, IndexError) as e:
        print(
            f"Invalid transcription model choice. Defaulting to {DEFAULT_TRANSCRIPTION_MODEL}"
        )

    return model


def main() -> None:
    """"""
    try:
        num_speakers_input: str = get_str_input(message="How many speakers?")

        if not num_speakers_input.isdigit():
            raise ValueError(
                f"Invalid number of speakers: {num_speakers_input}. Value must be an integer."
            )

        num_speakers: int = int(num_speakers_input)

        file_path_input: str = get_str_input(message="Enter file path")

        if not validate_file_path(file_path_input):
            raise ValueError(f"Invalid file path: {file_path_input}")

        file_path: str = file_path_input
        selected_model: str = select_transcription_model()

        print(f"[*] Loading transcription model '{selected_model}'...")

        transcription_model: FasterWhisperPipeline = whisperx.load_model(
            selected_model, device=DEVICE, compute_type=COMPUTE_TYPE
        )

        print("[OK] Transcription model successfully loaded")

        audio: np.ndarray = whisperx.load_audio(file_path)

        print("[OK] Audio successfully loaded")
        print("[*] Transcribing...")

        transcription: TranscriptionResult = transcription_model.transcribe(
            audio, batch_size=BATCH_SIZE
        )

        print("[OK] Transcription complete")
        print("[*] Loading alignment model...")

        align_model, metadata = whisperx.load_align_model(
            language_code=transcription["language"], device=DEVICE
        )

        print("[OK] Alignment model loaded")
        print("[*] Aligning...")

        aligned_transcription: dict[str, Any] = whisperx.align(
            transcription["segments"],
            align_model,
            metadata,
            audio,
            DEVICE,
        )

        print("[OK] Transcription and audio alignment complete")
        print("[*] Loading diarization model...")

        diarize_model: DiarizationPipeline = DiarizationPipeline(
            token=HF_TOKEN, device=DEVICE
        )

        print("[OK] Diarization model loaded")
        print("[*] Performing diarization...")

        segments: pd.DataFrame = cast(
            pd.DataFrame, diarize_model(audio=audio, num_speakers=num_speakers)
        )

        print("[OK] Diarization complete")
        print("[*] Assigning speakers to segments...")

        result = whisperx.assign_word_speakers(segments, aligned_transcription)

        print("[OK] Speaker segments successfully assigned")

        print(type(result))

    except ValueError as e:
        print(f"[ERROR] ValueError: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    main()
