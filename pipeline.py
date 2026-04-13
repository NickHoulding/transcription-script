"""Transcription pipeline class using WhisperX for ASR, alignment, and speaker diarization."""

import json
import logging
import sys
import warnings

logging.getLogger("whisperx").setLevel(logging.ERROR)
logging.getLogger("whisperx.vads.pyannote").setLevel(logging.ERROR)
logging.getLogger("whisperx.diarize").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.migration.utils").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", module="whisperx")
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import whisperx
from whisperx.asr import FasterWhisperPipeline, TranscriptionResult
from whisperx.diarize import DiarizationPipeline

from config import Config


class TranscriptionPipeline:
    """Runs the full WhisperX transcription, alignment, and diarization pipeline.

    Accepts pre-validated inputs and executes each processing stage in sequence,
    writing a TXT and JSON transcript to the specified save directory.
    """

    def __init__(
        self,
        file_path: str,
        save_path: str,
        num_speakers: int,
        model: str,
    ) -> None:
        """Initialise the pipeline with pre-validated run parameters.

        Args:
            file_path: Absolute path to the input audio file.
            save_path: Absolute path to the directory where output files are written.
            num_speakers: Expected number of speakers in the audio.
            model: WhisperX model name to use for transcription.
        """
        self._file_path = file_path
        self._save_path = save_path
        self._num_speakers = num_speakers
        self._model = model

    # -------------------------------------------------------------------------
    # File creation helpers
    # -------------------------------------------------------------------------

    def _write_txt(self, file_name: str, result: dict[str, Any]) -> None:
        """Write the speaker-labelled transcript to a .txt file.

        Args:
            file_name: Output filename stem (no extension).
            result: WhisperX result dict containing a ``"segments"`` list.

        Raises:
            OSError: If the file cannot be created or written.
        """
        file_path: Path = Path(self._save_path) / f"{file_name}_transcription.txt"

        with open(file_path, "w") as f:
            for segment in result["segments"]:
                speaker: str = segment.get("speaker", "UNKNOWN")
                text: str = segment.get("text", "")
                f.write(f"{speaker}: {text.strip()}\n")

    def _write_json(self, file_name: str, result: dict[str, Any]) -> None:
        """Write the full WhisperX result to a .json file.

        Args:
            file_name: Output filename stem (no extension).
            result: WhisperX result dict to serialise.

        Raises:
            OSError: If the file cannot be created or written.
        """
        file_path: Path = Path(self._save_path) / f"{file_name}_transcription.json"

        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Run the full transcription, alignment, and diarization pipeline."""
        try:
            print(f"[*] Loading transcription model '{self._model}'...")
            try:
                transcription_model: FasterWhisperPipeline = whisperx.load_model(
                    self._model, device=Config.device, compute_type=Config.compute_type
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load transcription model '{self._model}': {e}"
                ) from e

            print("[OK] Transcription model successfully loaded.")
            print("[*] Loading audio...")

            try:
                audio: np.ndarray = whisperx.load_audio(self._file_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load audio from '{self._file_path}': {e}"
                ) from e

            print("[OK] Audio successfully loaded.")

            print("[*] Transcribing...")
            try:
                transcription: TranscriptionResult = transcription_model.transcribe(
                    audio, batch_size=Config.batch_size
                )
            except Exception as e:
                raise RuntimeError(f"Transcription failed: {e}") from e

            print("[OK] Transcription complete.")

            print("[*] Loading alignment model...")
            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=transcription["language"], device=Config.device
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load alignment model: {e}") from e

            print("[OK] Alignment model loaded.")
            print("[*] Aligning audio segments...")
            try:
                aligned_transcription: dict[str, Any] = whisperx.align(
                    transcription["segments"],
                    align_model,
                    metadata,
                    audio,
                    Config.device,
                )
            except Exception as e:
                raise RuntimeError(f"Audio alignment failed: {e}") from e

            print("[OK] Audio alignment complete.")

            print("[*] Loading diarization model...")
            try:
                diarize_model: DiarizationPipeline = DiarizationPipeline(
                    token=Config.hf_token, device=Config.device
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load diarization model: {e}") from e

            print("[OK] Diarization model loaded.")
            print("[*] Performing diarization...")
            try:
                segments: pd.DataFrame = cast(
                    pd.DataFrame,
                    diarize_model(audio=audio, num_speakers=self._num_speakers),
                )
            except Exception as e:
                raise RuntimeError(f"Diarization failed: {e}") from e

            print("[OK] Diarization complete.")

            print("[*] Assigning speakers to segments...")
            try:
                result: dict[str, Any] = whisperx.assign_word_speakers(
                    segments, aligned_transcription
                )
            except Exception as e:
                raise RuntimeError(f"Speaker assignment failed: {e}") from e

            print("[OK] Speaker segments successfully assigned.")

            try:
                print("[*] Writing TXT file...")

                save_file_name: str = Path(self._file_path).stem
                self._write_txt(save_file_name, result=result)

                print(f"[OK] TXT successfully written to '{self._save_path}'.")
                print("[*] Writing JSON file...")

                self._write_json(save_file_name, result=result)

                print(f"[OK] JSON file successfully written to '{self._save_path}'.")
            except OSError as e:
                raise RuntimeError(
                    f"Failed to write output files to '{self._save_path}': {e}"
                ) from e

        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
            sys.exit(1)
