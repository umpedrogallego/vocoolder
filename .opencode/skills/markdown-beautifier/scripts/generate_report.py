#!/usr/bin/env python3
"""
Generate a formatted Markdown report from a text or markdown input.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a formatted markdown report")
    parser.add_argument("--input", required=True, help="Input text or markdown file")
    parser.add_argument("--output", required=True, help="Output markdown file")
    parser.add_argument("--style", default="technical", choices=["technical", "summary", "podcast"], help="Report style")
    parser.add_argument("--title", default=None, help="Optional report title")
    return parser.parse_args()


def summarize(lines: list[str], max_items: int) -> list[str]:
    items: list[str] = []
    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) > 140:
            clean = clean[:137] + "..."
        items.append(clean)
        if len(items) >= max_items:
            break
    return items


def build_report(text: str, title: str, style: str) -> str:
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]

    summary_items = summarize(non_empty, 3)
    key_points = summarize(non_empty, 5)
    outline = summarize(non_empty, 8)

    word_count = len(re.findall(r"\w+", text))
    line_count = len(lines)
    char_count = len(text)

    out: list[str] = []
    out.append(f"# {title}")
    out.append("")

    if style == "technical":
        out.append("## Summary")
        if summary_items:
            out.extend([f"- {item}" for item in summary_items])
        else:
            out.append("- No summary available")
        out.append("")
        out.append("## Metrics")
        out.append("")
        out.append("| Metric | Value |")
        out.append("|:-------|------:|")
        out.append(f"| Lines | {line_count} |")
        out.append(f"| Words | {word_count} |")
        out.append(f"| Characters | {char_count} |")
        out.append("")
        out.append("## Content")
        out.append("")
        out.append("```text")
        out.append(text.rstrip())
        out.append("```")

    elif style == "summary":
        out.append("## Key Points")
        if key_points:
            out.extend([f"- {item}" for item in key_points])
        else:
            out.append("- No key points found")
        out.append("")
        out.append("## Details")
        out.append("")
        out.append("```text")
        out.append(text.rstrip())
        out.append("```")

    elif style == "podcast":
        out.append("## Episode Outline")
        if outline:
            out.extend([f"- {item}" for item in outline])
        else:
            out.append("- No outline generated")
        out.append("")
        out.append("## Raw Notes")
        out.append("")
        out.append("```text")
        out.append(text.rstrip())
        out.append("```")

    out.append("")
    out.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    out.append("")

    return "\n".join(out)


def main() -> int:
    args = parse_args()
    title = args.title or os.path.splitext(os.path.basename(args.input))[0].replace("_", " ").title()

    with open(args.input, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    report = build_report(text, title, args.style)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
