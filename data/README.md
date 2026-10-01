# Data Directory

This directory stores evaluation manifests, sample audio fixtures, and cached embeddings.

## Structure
- `manifests/`: CSV and Parquet evaluation manifests containing audio paths, texts, languages, and human MOS ratings.
- `cache/`: SHA-256 keyed cache files for ASR transcriptions and neural speaker/emotion embeddings (gitignored).

## Manifest Specification
CSV/Parquet columns:
- `audio_path` (required): Relative or absolute path to target synthesized audio.
- `text` (required): Text prompt or gold transcript.
- `language` (required): ISO-639-1 code (e.g. `en`, `es`, `zh`, `hi`, `fr`, `ja`).
- `ref_audio_path` (optional): Reference speaker or style audio.
- `emotion` (optional): Target emotion category (`neutral`, `happy`, `sad`, `angry`, `surprised`).
- `system_id` (optional): Identifier of the generating TTS system.
- `human_mos` (optional): Ground-truth Mean Opinion Score from human listening tests.
