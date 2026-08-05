"""Run the hosted-model NuclearBench comparison suite."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from nuclearbench.constants import BenchmarkMode, DEFAULT_LOG_ROOT, DEFAULT_RESULTS_DIR
from nuclearbench.comparison import load_summaries, write_comparison_report
from nuclearbench.dataset import load_cases
from nuclearbench.model_adapter import (
    AnthropicMessagesAdapter,
    ModelAdapter,
    OpenAIResponsesAdapter,
    XAIResponsesAdapter,
)
from nuclearbench.run_evaluation import run_evaluation


@dataclass(frozen=True)
class HostedModelSpec:
    provider: str
    label: str
    model: str


DEFAULT_HOSTED_MODELS = (
    HostedModelSpec("openai", "ChatGPT", "gpt-5.4"),
    HostedModelSpec("xai", "Grok", "grok-4.20-reasoning"),
    HostedModelSpec("anthropic", "Claude", "claude-opus-4-7"),
)


def run_hosted_suite(
    run_id: str,
    mode: BenchmarkMode | str = BenchmarkMode.FORCED_CHOICE,
    dataset_path: str | None = None,
    case_ids: list[str] | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    models: tuple[HostedModelSpec, ...] = DEFAULT_HOSTED_MODELS,
) -> list[Path]:
    cases = load_cases(dataset_path)
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case.case_id in wanted]
        missing = wanted - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"Unknown case_ids: {', '.join(sorted(missing))}")

    outputs: list[Path] = []
    for spec in models:
        outputs.append(
            run_evaluation(
                adapter=_make_adapter(spec),
                cases=cases,
                mode=mode,
                run_id=run_id,
                log_root=log_root,
                results_dir=results_dir,
            )
        )
    return outputs


def _make_adapter(spec: HostedModelSpec) -> ModelAdapter:
    if spec.provider == "openai":
        return OpenAIResponsesAdapter(spec.model)
    if spec.provider == "xai":
        return XAIResponsesAdapter(spec.model)
    if spec.provider == "anthropic":
        return AnthropicMessagesAdapter(spec.model)
    raise ValueError(f"Unsupported provider: {spec.provider}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NuclearBench against hosted frontier models.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in BenchmarkMode],
        default=BenchmarkMode.FORCED_CHOICE.value,
    )
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--case_ids", nargs="*", default=None)
    parser.add_argument("--log_root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--openai_model", default=DEFAULT_HOSTED_MODELS[0].model)
    parser.add_argument("--xai_model", default=DEFAULT_HOSTED_MODELS[1].model)
    parser.add_argument("--anthropic_model", default=DEFAULT_HOSTED_MODELS[2].model)
    parser.add_argument(
        "--comparison_report",
        default=None,
        help="Optional output HTML path for a Plotly multi-model comparison report.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    models = (
        HostedModelSpec("openai", "ChatGPT", args.openai_model),
        HostedModelSpec("xai", "Grok", args.xai_model),
        HostedModelSpec("anthropic", "Claude", args.anthropic_model),
    )
    paths = run_hosted_suite(
        run_id=args.run_id,
        mode=args.mode,
        dataset_path=args.dataset_path,
        case_ids=args.case_ids,
        log_root=args.log_root,
        results_dir=args.results_dir,
        models=models,
    )
    if args.comparison_report:
        write_comparison_report(load_summaries(paths), args.comparison_report)
    for path in paths:
        print(path)
    if args.comparison_report:
        print(args.comparison_report)


if __name__ == "__main__":
    main()
