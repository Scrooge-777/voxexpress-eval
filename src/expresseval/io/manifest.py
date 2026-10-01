"""
Manifest parser supporting CSV and Parquet formats.
"""

from dataclasses import dataclass
import os
from typing import List, Optional
import pandas as pd


@dataclass
class ManifestItem:
    audio_path: str
    text: str
    language: str
    ref_audio_path: Optional[str] = None
    emotion: Optional[str] = None
    system_id: Optional[str] = None
    human_mos: Optional[float] = None
    id: Optional[str] = None


def load_manifest(file_path: str) -> List[ManifestItem]:
    """Loads evaluation items from a CSV or Parquet manifest."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Manifest file not found: {file_path}")

    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    # Validate essential columns
    required = ["audio_path", "text", "language"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    items = []
    for idx, row in df.iterrows():
        item = ManifestItem(
            audio_path=str(row["audio_path"]),
            text=str(row["text"]),
            language=str(row["language"]),
            ref_audio_path=str(row["ref_audio_path"]) if "ref_audio_path" in df.columns and pd.notna(row["ref_audio_path"]) else None,
            emotion=str(row["emotion"]) if "emotion" in df.columns and pd.notna(row["emotion"]) else None,
            system_id=str(row["system_id"]) if "system_id" in df.columns and pd.notna(row["system_id"]) else None,
            human_mos=float(row["human_mos"]) if "human_mos" in df.columns and pd.notna(row["human_mos"]) else None,
            id=str(row["id"]) if "id" in df.columns and pd.notna(row["id"]) else f"item_{idx}",
        )
        items.append(item)

    return items


def save_manifest(items: List[ManifestItem], output_path: str) -> None:
    """Saves a list of ManifestItem objects to CSV or Parquet."""
    records = []
    for item in items:
        records.append({
            "id": item.id,
            "audio_path": item.audio_path,
            "text": item.text,
            "language": item.language,
            "ref_audio_path": item.ref_audio_path,
            "emotion": item.emotion,
            "system_id": item.system_id,
            "human_mos": item.human_mos,
        })
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
