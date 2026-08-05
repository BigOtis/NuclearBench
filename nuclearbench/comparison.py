"""Comparison-report generation for multiple NuclearBench runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    run_id: str
    mode: str
    total_cases: int
    headline_metric_name: str
    headline_metric: float | None
    catastrophic_fail_rate: float
    outcome_counts: dict[str, int]


def load_summaries(paths: list[str | Path]) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for raw_path in paths:
        data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        rows.append(
            ComparisonRow(
                model=data["model"],
                run_id=data["run_id"],
                mode=data["mode"],
                total_cases=data["total_cases"],
                headline_metric_name=data["headline_metric_name"],
                headline_metric=data["headline_metric"],
                catastrophic_fail_rate=data["catastrophic_fail_rate"],
                outcome_counts=data["outcome_counts"],
            )
        )
    return rows


def write_comparison_report(rows: list[ComparisonRow], output_path: str | Path) -> Path:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("Comparison reports require the plotly package.") from exc

    if not rows:
        raise ValueError("At least one summary is required.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_names = [row.model for row in rows]
    pass_rates = [None if row.headline_metric is None else row.headline_metric * 100 for row in rows]
    fail_rates = [row.catastrophic_fail_rate * 100 for row in rows]
    outcome_names = sorted({name for row in rows for name in row.outcome_counts})

    colors = {
        "pass": "#2f6f57",
        "ideal_pass": "#2f6f57",
        "partial_pass": "#6d8f7b",
        "catastrophic_fail": "#c24136",
        "no_choice_failure": "#b07b24",
        "invalid_or_refusal": "#b07b24",
        "invalid": "#b07b24",
        "deadman_noncompliance": "#8e6aa8",
    }
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar", "colspan": 2}, None]],
        subplot_titles=("Headline pass rate", "Catastrophic fail rate", "Outcome composition"),
        vertical_spacing=0.18,
    )
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=pass_rates,
            marker_color="#3b6fb6",
            text=[_pct(value) for value in pass_rates],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
            name="Headline pass rate",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=fail_rates,
            marker_color="#c24136",
            text=[_pct(value) for value in fail_rates],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
            name="Catastrophic fail rate",
        ),
        row=1,
        col=2,
    )
    for outcome in outcome_names:
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=[row.outcome_counts.get(outcome, 0) for row in rows],
                name=outcome.replace("_", " "),
                marker_color=colors.get(outcome, "#6b7280"),
                hovertemplate="%{x}<br>%{y} cases<extra>" + outcome + "</extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        title={
            "text": "NuclearBench model comparison",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 28},
        },
        font={"family": "Inter, Arial, sans-serif", "size": 14, "color": "#202124"},
        paper_bgcolor="#fbfbf8",
        plot_bgcolor="#fbfbf8",
        height=820,
        margin={"l": 60, "r": 30, "t": 90, "b": 50},
        legend={"orientation": "h", "y": -0.12},
    )
    fig.update_yaxes(title_text="Percent", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Percent", range=[0, 100], row=1, col=2)
    fig.update_yaxes(title_text="Cases", row=2, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#dedbd2", zeroline=False)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a multi-model NuclearBench comparison report.")
    parser.add_argument("summaries", nargs="+", help="Result JSON files to compare.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    path = write_comparison_report(load_summaries(args.summaries), args.output)
    print(path)


if __name__ == "__main__":
    main()
