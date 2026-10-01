"""
Audio loading, caching, and manifest handling utilities.
"""

from expresseval.io.audio import AudioFile, load_audio, compute_file_hash
from expresseval.io.manifest import ManifestItem, load_manifest, save_manifest

__all__ = [
    "AudioFile",
    "load_audio",
    "compute_file_hash",
    "ManifestItem",
    "load_manifest",
    "save_manifest",
]
