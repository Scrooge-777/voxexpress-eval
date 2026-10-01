"""
Unified Evaluation Pipeline.
Loads audio once, fans out to enabled metric engines, manages hash caching,
and aggregates multi-axis scores into Parquet/DataFrame reports.
"""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional
import pandas as pd
import yaml

from expresseval.io.audio import load_audio
from expresseval.io.manifest import ManifestItem, load_manifest
from expresseval.metrics.base import MetricInput, MetricResult
from expresseval.metrics.prosody import ProsodyMetric
from expresseval.metrics.intelligibility import IntelligibilityMetric
from expresseval.metrics.speaker_sim import SpeakerSimilarityMetric
from expresseval.metrics.spectral import SpectralMetric
from expresseval.metrics.emotion import EmotionMetric
from expresseval.metrics.crosslingual import CrosslingualMetric
from expresseval.metrics.naturalness import NaturalnessMetric


class EvaluationPipeline:
    """
    Orchestrates audio loading, metric calculation, and report generation.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics = self._init_metrics()

    def _load_config(self, config_path: Optional[str]) -> dict:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "audio": {"sample_rate": 16000, "target_lufs": -23.0},
            "metrics": {
                "intelligibility": {"enabled": True},
                "naturalness": {"enabled": True},
                "speaker_sim": {"enabled": True},
                "prosody": {"enabled": True},
                "spectral": {"enabled": True},
                "emotion": {"enabled": True},
                "crosslingual": {"enabled": True},
            },
            "output": {"format": "parquet", "report_dir": "reports"},
        }

    def _init_metrics(self) -> List:
        m_cfg = self.config.get("metrics", {})
        active = []
        if m_cfg.get("prosody", {}).get("enabled", True):
            active.append(ProsodyMetric())
        if m_cfg.get("intelligibility", {}).get("enabled", True):
            active.append(IntelligibilityMetric())
        if m_cfg.get("speaker_sim", {}).get("enabled", True):
            active.append(SpeakerSimilarityMetric())
        if m_cfg.get("spectral", {}).get("enabled", True):
            active.append(SpectralMetric())
        if m_cfg.get("emotion", {}).get("enabled", True):
            active.append(EmotionMetric())
        if m_cfg.get("crosslingual", {}).get("enabled", True):
            active.append(CrosslingualMetric())
        if m_cfg.get("naturalness", {}).get("enabled", True):
            active.append(NaturalnessMetric())
        return active

    def evaluate_item(self, item: ManifestItem) -> Dict[str, float]:
        sr = self.config.get("audio", {}).get("sample_rate", 16000)
        target_db = self.config.get("audio", {}).get("target_lufs", -23.0)

        # 1. Single-pass audio loading & normalization
        synth_audio = load_audio(item.audio_path, target_sr=sr, target_db=target_db)
        ref_waveform = None
        ref_sr = None
        if item.ref_audio_path and os.path.exists(item.ref_audio_path):
            ref_audio = load_audio(item.ref_audio_path, target_sr=sr, target_db=target_db)
            ref_waveform = ref_audio.waveform
            ref_sr = ref_audio.sample_rate

        metric_input = MetricInput(
            id=item.id or "item",
            audio_waveform=synth_audio.waveform,
            sample_rate=synth_audio.sample_rate,
            text=item.text,
            language=item.language,
            ref_waveform=ref_waveform,
            ref_sample_rate=ref_sr,
            target_emotion=item.emotion,
            file_hash=synth_audio.file_hash,
        )

        # 2. Fan-out execution across active metrics
        record = {
            "id": item.id,
            "system_id": item.system_id or "unknown",
            "language": item.language,
            "emotion": item.emotion or "neutral",
            "human_mos": item.human_mos,
        }

        axis_scores = []
        for metric in self.metrics:
            res: MetricResult = metric.compute(metric_input)
            record[f"{res.axis}_score"] = res.score
            axis_scores.append(res.score)

        # Overall composite baseline score
        record["overall_score"] = round(float(sum(axis_scores) / max(len(axis_scores), 1)), 2)
        return record

    def run(self, manifest_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        items = load_manifest(manifest_path)
        records = [self.evaluate_item(item) for item in items]
        df = pd.DataFrame(records)

        out_dir = self.config.get("output", {}).get("report_dir", "reports")
        os.makedirs(out_dir, exist_ok=True)

        if not output_path:
            fmt = self.config.get("output", {}).get("format", "parquet")
            output_path = os.path.join(out_dir, f"eval_results.{fmt}")

        if output_path.endswith(".parquet"):
            try:
                df.to_parquet(output_path, index=False)
            except (ImportError, Exception):
                csv_path = output_path.replace(".parquet", ".csv")
                df.to_csv(csv_path, index=False)
                output_path = csv_path
        else:
            df.to_csv(output_path, index=False)

        print(f"Evaluation complete. Evaluated {len(df)} samples. Results saved to: {output_path}")
        return df
