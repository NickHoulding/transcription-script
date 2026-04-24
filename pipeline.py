"""Transcription pipeline class using WhisperX for ASR, alignment, and speaker diarization."""

import json
import logging
import sys
import time
import warnings
from collections.abc import Generator
from contextlib import contextmanager

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
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from whisperx.asr import FasterWhisperPipeline, TranscriptionResult
from whisperx.diarize import DiarizationPipeline

from config import Config


def _format_elapsed_time(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A string like ``4.3s``, ``2m 07s``, or ``1h 02m 07s``, using the
        smallest unit combination that avoids leading zero components.
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes:
        return f"{minutes}m {seconds:02d}s"
    else:
        return f"{seconds:.1f}s"


@contextmanager
def _spinner(label: str) -> Generator[None, None, None]:
    """Context manager that shows an animated spinner while a block executes.

    Prints the label and elapsed time to stdout when the block exits.

    Args:
        label: Text displayed next to the spinner during execution.
    """
    start_time: float = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn(label),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task("", total=None)
        yield

    elapsed_time: float = time.monotonic() - start_time
    print(f"{label} ({_format_elapsed_time(elapsed_time)})")


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
    # Pipeline step helpers
    # -------------------------------------------------------------------------

    def _load_transcription_model(self) -> FasterWhisperPipeline:
        """Load the WhisperX transcription model.

        Returns:
            The loaded transcription model.

        Raises:
            RuntimeError: If the model fails to load.
        """
        with _spinner(f"Loading transcription model '{self._model}'"):
            try:
                model: FasterWhisperPipeline = whisperx.load_model(
                    self._model, device=Config.device, compute_type=Config.compute_type
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load transcription model '{self._model}': {e}"
                ) from e
        return model

    def _load_audio(self) -> np.ndarray:
        """Load the audio file from disk.

        Returns:
            The audio waveform as a NumPy array.

        Raises:
            RuntimeError: If the audio file cannot be loaded.
        """
        with _spinner("Loading audio"):
            try:
                audio: np.ndarray = whisperx.load_audio(self._file_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load audio from '{self._file_path}': {e}"
                ) from e
        return audio

    def _transcribe(
        self, model: FasterWhisperPipeline, audio: np.ndarray
    ) -> TranscriptionResult:
        """Transcribe the audio using the loaded model.

        Args:
            model: The loaded WhisperX transcription model.
            audio: The audio waveform to transcribe.

        Returns:
            The raw transcription result.

        Raises:
            RuntimeError: If transcription fails.
        """
        with _spinner("Transcribing"):
            try:
                transcription: TranscriptionResult = model.transcribe(
                    audio, batch_size=Config.batch_size
                )
            except Exception as e:
                raise RuntimeError(f"Transcription failed: {e}") from e
        return transcription

    def _align(
        self, transcription: TranscriptionResult, audio: np.ndarray
    ) -> dict[str, Any]:
        """Load the alignment model and align transcription segments to the audio.

        Args:
            transcription: The raw transcription result containing segments and language.
            audio: The audio waveform used during transcription.

        Returns:
            The aligned transcription result.

        Raises:
            RuntimeError: If the alignment model fails to load or alignment fails.
        """
        with _spinner("Loading alignment model"):
            try:
                align_model, metadata = whisperx.load_align_model(
                    language_code=transcription["language"], device=Config.device
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load alignment model: {e}") from e

        with _spinner("Aligning audio segments"):
            try:
                aligned: dict[str, Any] = whisperx.align(
                    transcription["segments"],
                    align_model,
                    metadata,
                    audio,
                    Config.device,
                )
            except Exception as e:
                raise RuntimeError(f"Audio alignment failed: {e}") from e
        return aligned

    def _diarize(self, audio: np.ndarray) -> pd.DataFrame:
        """Load the diarization model and assign speaker segments to the audio.

        Args:
            audio: The audio waveform to diarize.

        Returns:
            A DataFrame of speaker-labelled time segments.

        Raises:
            RuntimeError: If the diarization model fails to load or diarization fails.
        """
        with _spinner("Loading diarization model"):
            try:
                diarize_model: DiarizationPipeline = DiarizationPipeline(
                    token=Config.hf_token, device=Config.device
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load diarization model: {e}") from e

        with _spinner("Performing diarization"):
            try:
                segments: pd.DataFrame = cast(
                    pd.DataFrame,
                    diarize_model(audio=audio, num_speakers=self._num_speakers),
                )
            except Exception as e:
                raise RuntimeError(f"Diarization failed: {e}") from e
        return segments

    def _assign_speakers(
        self, segments: pd.DataFrame, aligned_transcription: dict[str, Any]
    ) -> dict[str, Any]:
        """Assign speaker labels to aligned transcription segments.

        Args:
            segments: Diarization output mapping time ranges to speaker labels.
            aligned_transcription: The aligned transcription result.

        Returns:
            The final result dict with speaker labels attached to each segment.

        Raises:
            RuntimeError: If speaker assignment fails.
        """
        with _spinner("Assigning speakers to segments"):
            try:
                result: dict[str, Any] = whisperx.assign_word_speakers(
                    segments, aligned_transcription
                )
            except Exception as e:
                raise RuntimeError(f"Speaker assignment failed: {e}") from e
        return result

    def _write_output(self, result: dict[str, Any]) -> None:
        """Write the final transcript to TXT and JSON output files.

        Args:
            result: The fully processed WhisperX result dict.

        Raises:
            RuntimeError: If either output file cannot be written.
        """
        save_file_name: str = Path(self._file_path).stem
        try:
            with _spinner("Writing TXT file"):
                self._write_txt(save_file_name, result=result)

            with _spinner("Writing JSON file"):
                self._write_json(save_file_name, result=result)
        except OSError as e:
            raise RuntimeError(
                f"Failed to write output files to '{self._save_path}': {e}"
            ) from e

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Run the full transcription, alignment, and diarization pipeline."""
        try:
            start_time: float = time.monotonic()

            transcription_model = self._load_transcription_model()
            audio = self._load_audio()
            transcription = self._transcribe(transcription_model, audio)
            aligned_transcription = self._align(transcription, audio)
            segments = self._diarize(audio)
            result = self._assign_speakers(segments, aligned_transcription)
            self._write_output(result)

            print(
                f"Total elapsed time ({_format_elapsed_time(time.monotonic() - start_time)})"
            )
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {type(e).__name__}: {e}")
            sys.exit(1)
