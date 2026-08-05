import json
import xml.etree.ElementTree as ET

from nuclearbench.compile_report import compile_report


def test_compile_report_writes_blog_tables_and_valid_svg(tmp_path):
    run_id = "unit-report"
    results = tmp_path / "results"
    run_dir = tmp_path / "run"
    output = tmp_path / "final"
    results.mkdir()
    run_dir.mkdir()
    summary = {
        "run_id": run_id,
        "model": "small-model",
        "mode": "forced_choice",
        "total_cases": 2,
        "outcome_counts": {"pass": 1, "no_choice_failure": 1},
        "case_ids_by_outcome": {"pass": ["NB-001"], "no_choice_failure": ["NB-002"]},
    }
    (results / f"small-model.{run_id}.forced_choice.json").write_text(json.dumps(summary))
    manifest = {
        "models": [{"label": "small-model"}, {"label": "broken-model"}],
        "case_ids": ["NB-001", "NB-002"],
        "results": [
            {"label": "small-model", "status": "completed"},
            {"label": "broken-model", "status": "failed", "error": "load error"},
        ],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))

    blog = compile_report(run_id, results, run_dir, output)

    text = blog.read_text(encoding="utf-8")
    assert "1 of 2 requested models completed" in text
    assert "broken-model" in text
    assert (output / "results.csv").is_file()
    assert (output / "report.json").is_file()
    ET.parse(output / "charts" / "outcomes.svg")
    ET.parse(output / "charts" / "pass_rates.svg")
