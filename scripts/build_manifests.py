#!/usr/bin/env python3
"""
Manifest builder utility.
Scans audio directories and generates evaluation manifests (CSV / Parquet).
"""

import argparse
import os
import pandas as pd


def build_manifest(audio_dir: str, output_path: str, default_lang: str = "en"):
    records = []
    supported_exts = {".wav", ".mp3", ".flac", ".ogg"}

    if not os.path.exists(audio_dir):
        print(f"Directory not found: {audio_dir}")
        return

    for root, _, files in os.walk(audio_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, start=os.getcwd())
                sample_id = os.path.splitext(f)[0]
                records.append({
                    "id": sample_id,
                    "audio_path": rel_path.replace("\\", "/"),
                    "text": "Sample utterance transcript for evaluation.",
                    "language": default_lang,
                    "ref_audio_path": None,
                    "emotion": "neutral",
                    "system_id": "baseline_system",
                    "human_mos": None,
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"Generated manifest with {len(df)} entries at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build ExpressEval dataset manifest")
    parser.add_argument("--audio-dir", "-d", required=True, help="Directory containing audio files")
    parser.add_argument("--output", "-o", default="data/manifests/manifest.csv", help="Output path for CSV/Parquet")
    parser.add_argument("--lang", "-l", default="en", help="Default language code")
    args = parser.parse_args()

    build_manifest(args.audio_dir, args.output, default_lang=args.lang)


if __name__ == "__main__":
    main()
