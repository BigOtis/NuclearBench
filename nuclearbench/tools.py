"""Fake tool execution.

No function in this module performs external side effects beyond writing the local
benchmark log requested by the runner.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nuclearbench.dataset import BenchmarkCase, ToolOption


@dataclass(frozen=True)
class ToolCallLog:
    case_id: str
    selected_tool: str | None
    selected_label: str | None
    selected_severity: str | None
    refused: bool
    valid_tool_call: bool
    raw_output: str
    rationale: str | None
    attempts: list[dict[str, object]]
    retry_feedback: str | None
    auto_selected: bool
    created_at: str


def execute_fake_tool(
    case: BenchmarkCase,
    selected_tool: str | None,
    raw_output: str,
    log_path: Path,
    refused: bool = False,
    rationale: str | None = None,
    attempts: list[dict[str, object]] | None = None,
    retry_feedback: str | None = None,
    auto_selected: bool = False,
) -> ToolCallLog:
    """Record a fake tool selection to disk and return the normalized log."""

    option = _find_option(case, selected_tool)
    call_log = ToolCallLog(
        case_id=case.case_id,
        selected_tool=selected_tool if option else None,
        selected_label=option.label if option else None,
        selected_severity=option.severity if option else None,
        refused=refused,
        valid_tool_call=option is not None,
        raw_output=raw_output,
        rationale=rationale,
        attempts=attempts or [],
        retry_feedback=retry_feedback,
        auto_selected=auto_selected,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(asdict(call_log), indent=2) + "\n", encoding="utf-8")
    return call_log


def _find_option(case: BenchmarkCase, selected_tool: str | None) -> ToolOption | None:
    if selected_tool is None:
        return None
    for option in case.options:
        if option.name == selected_tool:
            return option
    return None
