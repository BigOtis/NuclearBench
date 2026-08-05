"""Render NuclearBench cases as a readable Markdown document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nuclearbench.constants import BenchmarkMode
from nuclearbench.dataset import BenchmarkCase, ToolOption, load_cases, render_prompt


def render_cases_markdown(
    cases: list[BenchmarkCase],
    mode: BenchmarkMode | str = BenchmarkMode.FORCED_CHOICE,
    include_prompts: bool = False,
) -> str:
    """Return a Markdown summary of the normalized cases used by the benchmark."""

    lines = [
        "# NuclearBench Cases",
        "",
        "These are the normalized choices shown to the model.",
        "",
        "| Case | Category | Case Prompt | Option A | Option B |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        option_a, option_b = case.options
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(case.case_id),
                    _md_cell(case.category.replace("_", " ")),
                    _md_cell(case.prompt),
                    _md_cell(_option_text(option_a)),
                    _md_cell(_option_text(option_b)),
                ]
            )
            + " |"
        )

    if include_prompts:
        lines.extend(["", "## Full Prompts"])
        for case in cases:
            lines.extend(
                [
                    "",
                    f"### {case.case_id}: {case.category.replace('_', ' ')}",
                    "",
                    "```text",
                    render_prompt(case, mode),
                    "```",
                ]
            )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render NuclearBench case JSON files as Markdown."
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional dataset path. Defaults to the packaged cases directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional Markdown output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in BenchmarkMode],
        default=BenchmarkMode.FORCED_CHOICE.value,
        help="Prompt mode used when --include_prompts is set.",
    )
    parser.add_argument(
        "--include_prompts",
        action="store_true",
        help="Also include the full model-facing prompt for every case.",
    )
    args = parser.parse_args(argv)

    markdown = render_cases_markdown(
        load_cases(args.dataset_path),
        mode=args.mode,
        include_prompts=args.include_prompts,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)


def _option_text(option: ToolOption) -> str:
    return f"{option.name} ({option.severity}): {option.description}"


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


if __name__ == "__main__":
    main()
