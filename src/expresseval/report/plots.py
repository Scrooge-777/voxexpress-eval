"""
Radar and Pareto plotting utilities.
"""

from typing import Dict


def generate_ascii_radar(scores: Dict[str, float], bar_width: int = 25, use_ascii: bool = True) -> str:
    """
    Renders a clean ASCII horizontal radar chart for terminal and Markdown logs.
    """
    char_fill = "#" if use_ascii else "█"
    char_empty = "-" if use_ascii else "░"
    char_line = "-" if use_ascii else "─"
    lines = ["ExpressEval Metric Radar Breakdown:"]
    lines.append(char_line * 45)
    for axis, score in scores.items():
        clean_name = axis.replace("_score", "").capitalize()
        filled = int((score / 100.0) * bar_width)
        empty = bar_width - filled
        bar = char_fill * filled + char_empty * empty
        lines.append(f"{clean_name:18s} | {bar} | {score:5.1f} / 100")
    lines.append(char_line * 45)
    return "\n".join(lines)
