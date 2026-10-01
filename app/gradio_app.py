"""
ExpressEval Interactive Web Demo (Gradio / Hugging Face Spaces).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from expresseval.pipeline import EvaluationPipeline
from expresseval.io.manifest import ManifestItem
from expresseval.report.plots import generate_ascii_radar


def evaluate_sample(audio_path, text, language, ref_audio_path=None, emotion="neutral"):
    if audio_path is None or not text:
        return "Please upload an audio file and provide the text transcript.", ""

    pipeline = EvaluationPipeline()
    item = ManifestItem(
        audio_path=audio_path,
        text=text,
        language=language,
        ref_audio_path=ref_audio_path,
        emotion=emotion,
        id="demo_sample",
    )
    result = pipeline.evaluate_item(item)

    score_lines = [f"### 🏆 Overall Score: {result.get('overall_score', 0)} / 100\n"]
    axis_scores = {}
    for k, v in result.items():
        if k.endswith("_score") and k != "overall_score":
            axis_scores[k] = v
            clean_axis = k.replace("_score", "").capitalize()
            score_lines.append(f"- **{clean_axis}**: {v} / 100")

    radar_text = generate_ascii_radar(axis_scores)
    return "\n".join(score_lines), f"```text\n{radar_text}\n```"


def launch_app():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with `pip install gradio` to run the web demo.")
        return

    demo = gr.Interface(
        fn=evaluate_sample,
        inputs=[
            gr.Audio(type="filepath", label="Synthesized Audio (WAV / MP3 / FLAC)"),
            gr.Textbox(label="Text Transcript", placeholder="Enter the text that was spoken..."),
            gr.Dropdown(choices=["en", "es", "zh", "hi", "fr", "de", "ja"], value="en", label="Language"),
            gr.Audio(type="filepath", label="Optional Reference Audio (Speaker Voice)"),
            gr.Dropdown(choices=["neutral", "happy", "sad", "angry", "surprised"], value="neutral", label="Target Emotion"),
        ],
        outputs=[
            gr.Markdown(label="Evaluation Scores"),
            gr.Markdown(label="Diagnostic Radar Chart"),
        ],
        title="🎙️ ExpressEval: Speech Synthesis Quality Benchmark",
        description="Automatic objective evaluation of expressive, multilingual text-to-speech audio.",
    )
    demo.launch()


if __name__ == "__main__":
    launch_app()
