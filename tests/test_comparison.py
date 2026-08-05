import json

import pytest

from nuclearbench.comparison import load_summaries, write_comparison_report


def test_comparison_report_renders_multiple_models(tmp_path):
    pytest.importorskip("plotly")
    summaries = []
    for model, pass_rate, fail_rate in [
        ("model-a", 0.75, 0.25),
        ("model-b", 0.5, 0.5),
    ]:
        path = tmp_path / f"{model}.json"
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "run_id": "compare",
                    "mode": "forced_choice",
                    "total_cases": 4,
                    "headline_metric_name": "forced_choice_pass_rate",
                    "headline_metric": pass_rate,
                    "catastrophic_fail_rate": fail_rate,
                    "outcome_counts": {
                        "pass": int(pass_rate * 4),
                        "catastrophic_fail": int(fail_rate * 4),
                    },
                }
            )
        )
        summaries.append(path)

    output = write_comparison_report(load_summaries(summaries), tmp_path / "comparison.html")

    html = output.read_text()
    assert "NuclearBench model comparison" in html
    assert "model-a" in html
    assert "model-b" in html
    assert "catastrophic fail rate" in html.lower()
