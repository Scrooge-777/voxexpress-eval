"""
Command Line Interface for ExpressEval.
"""

import os
import sys
import click

from expresseval import __version__
from expresseval.pipeline import EvaluationPipeline
from expresseval.io.manifest import ManifestItem


@click.group()
@click.version_option(version=__version__, prog_name="expresseval")
def main():
    """ExpressEval: Automatic Objective Evaluation of Multilingual Expressive Speech Synthesis (TTS)."""
    pass


@main.command()
@click.option("--manifest", "-m", required=True, type=click.Path(exists=True), help="Path to evaluation manifest (CSV or Parquet).")
@click.option("--config", "-c", default=None, type=click.Path(exists=True), help="Path to pipeline configuration YAML.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Path to save evaluation results.")
def run(manifest, config, output):
    """Run full evaluation pipeline on a dataset manifest."""
    click.echo(f"Initializing ExpressEval Pipeline [v{__version__}]...")
    pipeline = EvaluationPipeline(config_path=config)
    df = pipeline.run(manifest_path=manifest, output_path=output)
    click.echo(f"Successfully evaluated {len(df)} utterances.")
    click.echo(df.describe().to_string())


@main.command()
@click.option("--audio", "-a", required=True, type=click.Path(exists=True), help="Path to synthesized audio file.")
@click.option("--text", "-t", required=True, type=str, help="Text transcript.")
@click.option("--lang", "-l", default="en", type=str, help="Language code (e.g. en, es, zh, hi, fr).")
@click.option("--ref", "-r", default=None, type=click.Path(exists=True), help="Optional reference speaker audio.")
@click.option("--emotion", "-e", default="neutral", type=str, help="Target emotion tag.")
def eval(audio, text, lang, ref, emotion):
    """Evaluate a single audio sample from the command line."""
    click.echo(f"Evaluating {audio} [{lang}]...")
    pipeline = EvaluationPipeline()
    item = ManifestItem(
        audio_path=audio,
        text=text,
        language=lang,
        ref_audio_path=ref,
        emotion=emotion,
        id="cli_sample",
    )
    result = pipeline.evaluate_item(item)
    click.echo("\n--- ExpressEval Diagnostic Score Card ---")
    for k, v in result.items():
        click.echo(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
