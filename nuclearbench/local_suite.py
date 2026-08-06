"""Run NuclearBench against a manifest or command-line list of local models."""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import snapshot_download

from nuclearbench.comparison import load_summaries, write_comparison_report
from nuclearbench.constants import BenchmarkMode, DEFAULT_LOG_ROOT, DEFAULT_RESULTS_DIR
from nuclearbench.dataset import load_cases
from nuclearbench.model_adapter import HuggingFaceCausalLMAdapter
from nuclearbench.run_evaluation import run_evaluation


DEFAULT_MODEL_MANIFEST = Path("models") / "local_models.json"


@dataclass(frozen=True)
class LocalModelSpec:
    label: str
    model: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    prompt_format: str = "auto"
    trust_remote_code: bool = False
    use_cache: bool = True
    enable_thinking: bool | None = None
    continuation_new_tokens: int | None = None
    released: str | None = None
    parameters: str | None = None
    rationale: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "LocalModelSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown model manifest keys: {', '.join(sorted(unknown))}")
        return cls(**value)


def load_model_manifest(path: str | Path = DEFAULT_MODEL_MANIFEST) -> tuple[LocalModelSpec, ...]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Model manifest must contain a non-empty model list: {manifest_path}")
    specs = tuple(LocalModelSpec.from_dict(row) for row in rows)
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Model labels must be unique.")
    return specs


def run_local_suite(
    run_id: str,
    mode: BenchmarkMode | str = BenchmarkMode.FORCED_CHOICE,
    dataset_path: str | None = None,
    case_ids: list[str] | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    report_dir: str | Path = Path("reports") / "current" / "runs",
    models: tuple[LocalModelSpec, ...] = (),
    device_map: str | None = "auto",
    local_files_only: bool = False,
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    download_only: bool = False,
    stop_on_error: bool = False,
    resume: bool = False,
    skip_models: set[str] | None = None,
    skip_reason: str = "explicitly skipped by driver",
) -> list[Path]:
    if not models:
        raise ValueError("At least one local model is required.")
    cases = load_cases(dataset_path)
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case.case_id in wanted]
        missing = wanted - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"Unknown case_ids: {', '.join(sorted(missing))}")

    output_dir = Path(report_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "mode": BenchmarkMode(mode).value,
        "case_ids": [case.case_id for case in cases],
        "quantization": "4bit" if load_in_4bit else "8bit" if load_in_8bit else "none",
        "models": [asdict(spec) for spec in models],
        "results": [],
    }
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, run_manifest)

    summaries: list[Path] = []
    for spec in models:
        print(f"Preparing {spec.label}: {spec.model}", flush=True)
        adapter = None
        status: dict[str, object] = {"label": spec.label, "model": spec.model}
        try:
            if skip_models and spec.label in skip_models:
                status.update(status="skipped", reason=skip_reason)
                print(f"Skipping {spec.label}", flush=True)
                continue
            existing = Path(results_dir) / f"{spec.label}.{run_id}.{BenchmarkMode(mode).value}.json"
            if resume and _summary_matches(existing, run_id, mode, {case.case_id for case in cases}):
                summaries.append(existing)
                status.update(status="reused", summary=str(existing))
                print(f"Reusing completed {spec.label}", flush=True)
                continue
            if download_only:
                snapshot_path = snapshot_download(spec.model)
                status.update(status="downloaded", snapshot_path=str(snapshot_path))
                print(f"Downloaded {spec.label} to {snapshot_path}", flush=True)
                continue
            adapter = HuggingFaceCausalLMAdapter(
                model_name_or_path=spec.model,
                max_new_tokens=spec.max_new_tokens,
                temperature=spec.temperature,
                top_p=spec.top_p,
                top_k=spec.top_k,
                device_map=device_map,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code or spec.trust_remote_code,
                prompt_format=spec.prompt_format,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                use_cache=spec.use_cache,
                enable_thinking=spec.enable_thinking,
                continuation_new_tokens=spec.continuation_new_tokens,
            )
            adapter.model_name = spec.label
            print(f"Running {spec.label} on {len(cases)} cases", flush=True)
            summary = run_evaluation(
                adapter=adapter,
                cases=cases,
                mode=mode,
                run_id=run_id,
                log_root=log_root,
                results_dir=results_dir,
                html_report=output_dir / f"{spec.label}.html",
                resume=resume,
            )
            summaries.append(summary)
            status.update(status="completed", summary=str(summary))
            print(f"Finished {spec.label}", flush=True)
        except Exception as exc:
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"Failed {spec.label}: {exc}", flush=True)
            if stop_on_error:
                raise
        finally:
            run_manifest["results"].append(status)
            _write_json(manifest_path, run_manifest)
            torch_module = getattr(adapter, "_torch", None) if adapter is not None else None
            if adapter is not None:
                model = getattr(adapter, "model", None)
                if model is not None and hasattr(model, "cpu"):
                    try:
                        model.cpu()
                    except Exception:
                        pass
                del adapter
            gc.collect()
            if torch_module is not None and torch_module.cuda.is_available():
                try:
                    torch_module.cuda.empty_cache()
                    torch_module.cuda.ipc_collect()
                except Exception as cleanup_exc:
                    print(f"CUDA cleanup warning after {spec.label}: {cleanup_exc}", flush=True)
                    # Reset the device context so later models can still run.
                    try:
                        torch_module.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
    run_manifest["finished_at"] = datetime.now(UTC).isoformat()
    _write_json(manifest_path, run_manifest)
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--report_dir", default=str(Path("reports") / "current" / "runs"))
    parser.add_argument("--model_manifest", default=str(DEFAULT_MODEL_MANIFEST))
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Manifest labels or arbitrary Hugging Face model IDs/local paths.",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "default", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse exact completed model summaries and per-case report.json logs.",
    )
    parser.add_argument("--skip_models", nargs="*", default=None, help="Manifest labels to record but not run.")
    parser.add_argument("--skip_reason", default="explicitly skipped by driver")
    parser.add_argument("--comparison_report", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = load_model_manifest(args.model_manifest)
    models = _select_models(manifest, args.models)
    paths = run_local_suite(
        run_id=args.run_id,
        mode=args.mode,
        dataset_path=args.dataset_path,
        case_ids=args.case_ids,
        log_root=args.log_root,
        results_dir=args.results_dir,
        report_dir=args.report_dir,
        models=models,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        download_only=args.download_only,
        stop_on_error=args.stop_on_error,
        resume=args.resume,
        skip_models=set(args.skip_models or []),
        skip_reason=args.skip_reason,
    )
    if args.comparison_report and paths:
        write_comparison_report(load_summaries(paths), args.comparison_report)
    for path in paths:
        print(path)
    if args.comparison_report and paths:
        print(args.comparison_report)


def _select_models(
    manifest: tuple[LocalModelSpec, ...], values: list[str] | None
) -> tuple[LocalModelSpec, ...]:
    if not values:
        return manifest
    by_label = {spec.label: spec for spec in manifest}
    selected = []
    for value in values:
        selected.append(by_label.get(value) or LocalModelSpec(label=_slug(value), model=value))
    return tuple(selected)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "model"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summary_matches(
    path: Path,
    run_id: str,
    mode: BenchmarkMode | str,
    case_ids: set[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = {
            case_id
            for values in payload.get("case_ids_by_outcome", {}).values()
            for case_id in values
        }
        return (
            payload.get("run_id") == run_id
            and payload.get("mode") == BenchmarkMode(mode).value
            and recorded == case_ids
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


if __name__ == "__main__":
    main()
