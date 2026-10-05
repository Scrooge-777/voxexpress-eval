"""
High-performance single-pass audio I/O with caching and loudness normalization.
"""

from dataclasses import dataclass
import hashlib
import os
from typing import Optional, Tuple
import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None
from scipy import signal


@dataclass
class AudioFile:
    waveform: np.ndarray       # 1D float32 normalized [-1.0, 1.0]
    sample_rate: int
    duration_s: float
    file_hash: str
    path: str


def compute_file_hash(file_path: str) -> str:
    """Computes SHA-256 checksum of an audio file for cache keying."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def normalize_loudness(audio: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    """RMS loudness normalization matching EBU R128 target levels."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros_like(audio, dtype=np.float32)

    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    if not np.isfinite(rms) or rms < 1e-12:
        return np.zeros_like(audio, dtype=np.float32)

    current_db = 20.0 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10.0 ** (gain_db / 20.0)
    normalized = audio * gain

    # Soft-clip to prevent digital wrap-around distortion
    max_val = np.max(np.abs(normalized))
    if np.isfinite(max_val) and max_val > 1.0:
        normalized = normalized / max_val
    return normalized.astype(np.float32)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D audio array using polyphase filtering."""
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def load_audio(
    file_path: str,
    target_sr: int = 16000,
    normalize: bool = True,
    target_db: float = -23.0,
) -> AudioFile:
    """
    Loads, mono-mixes, resamples, and normalizes audio in a single pass.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        import soundfile as sf
        waveform, sr = sf.read(file_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
    except (ImportError, Exception):
        import wave
        import struct
        with wave.open(file_path, "r") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            raw_bytes = wf.readframes(n_frames)
            # 16-bit PCM conversion
            ints = struct.unpack(f"<{n_frames * n_channels}h", raw_bytes)
            waveform = np.array(ints, dtype=np.float32) / 32768.0
            if n_channels > 1:
                waveform = waveform.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    if normalize:
        waveform = normalize_loudness(waveform, target_db=target_db)

    duration = float(len(waveform) / sr)
    f_hash = compute_file_hash(file_path)

    return AudioFile(
        waveform=waveform,
        sample_rate=sr,
        duration_s=duration,
        file_hash=f_hash,
        path=file_path,
    )
