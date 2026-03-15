""""""

import os
from typing import Any, cast

import numpy as np
import pandas as pd
import whisperx
from whisperx.asr import FasterWhisperPipeline, TranscriptionResult
from whisperx.diarize import DiarizationPipeline


# Global Configuration
COMPUTE_TYPE: str = "float16"
DEVICE: str = "cuda"
BATCH_SIZE: int = 16
HF_TOKEN: str = "INSERT_HF_TOKEN_HERE"
TRANSCRIPTION_MODELS: dict[str, str] = {
    "1": "tiny.en",
    "2": "base.en",
    "3": "small.en",
    "4": "medium.en",
    "5": "large-v2",
    "6": "large-v3",
    "7": "turbo"
}


def validate_file_path(file_path: str) -> bool:
    """"""
    return (
        len(file_path) > 0 
        and os.path.exists(file_path)
        and os.path.isfile(file_path)
    )


def select_transcription_model() -> str:
    """"""
    for number, model in enumerate(TRANSCRIPTION_MODELS):
        print(f"[{number}] {model}")

    selected_model: str = input(f"Select a model:\n>>> ")

    if selected_model not in TRANSCRIPTION_MODELS:
        raise ValueError(f"Invalid transcription model choice: {selected_model}")
    
    return TRANSCRIPTION_MODELS[selected_model]


def main() -> None:
    """"""
    try:
        input_val: str = input(f"How many speakers?:\n>>> ")

        if not input_val.isdigit():
            raise ValueError(f"Invalid number of speakers: {input_val}. Value must be an integer.")
        
        num_speakers: int = int(input_val)

        input_val: str = input(f"Enter file path:\n>>> ")

        if not validate_file_path(input_val):
            raise ValueError(f"Invalid file path: {input_val}")
        
        file_path: str = input_val
        
        print("[OK] Validated file path")
        print("[*] Loading transcription model...")

        selected_model: str = select_transcription_model()
        transcription_model: FasterWhisperPipeline = whisperx.load_model(selected_model)
        
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
            language_code=transcription["language"],
            device=DEVICE
        )

        print("[OK] Alignment model loaded")
        print("[*] Aligning...")

        aligned_transcription: dict[str, Any] = whisperx.align(
            transcription["segments"],
            align_model,
            metadata,
            audio,
            DEVICE,
            return_chat_alignments=False
        )

        print("[OK] Transcription and audio alignment complete")
        print("[*] Loading diarization model...")

        diarize_model: DiarizationPipeline = DiarizationPipeline(
            token=HF_TOKEN,
            device=DEVICE
        )

        print("[OK] Diarization model loaded")
        print("[*] Performing diarization...")

        segments: pd.DataFrame = cast(pd.DataFrame, diarize_model(
            audio=audio,
            num_speakers=num_speakers
        ))

        print("[OK] Diarization complete")
        print("[*] Assigning speakers to segments...")

        result = whisperx.assign_word_speakers(
            segments, aligned_transcription
        )
        
        print("[OK] Speaker segments successfully assigned")
        
        print(type(result))
        
    except ValueError as e:
        print(f"[ERROR] ValueError: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")


if __name__ == "__main__":
    main()
