import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[2] / "rlinf" / "utils" / "performance_measurement.py"
)
SPEC = importlib.util.spec_from_file_location("performance_measurement", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
record_event = MODULE.record_event
record_span = MODULE.record_span


def _records(path):
    files = list(path.glob("events-*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_record_event_is_disabled_without_output_env(monkeypatch, tmp_path):
    monkeypatch.delenv("RLINF_PERF_MEASUREMENT_DIR", raising=False)
    monkeypatch.delenv("RLINF_PERF_RUN_ID", raising=False)

    record_event(
        owner="runner",
        rank=0,
        outer_step=1,
        stage="step.resident_outer",
        event="start",
    )

    assert list(tmp_path.iterdir()) == []


def test_record_event_writes_monotonic_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("RLINF_PERF_MEASUREMENT_DIR", str(tmp_path))
    monkeypatch.setenv("RLINF_PERF_RUN_ID", "qualification-a")

    record_event(
        owner="rollout",
        rank=3,
        outer_step=2,
        stage="rollout.service",
        event="start",
        timestamp_ns=123,
        request_id=7,
        metadata={"pipeline_stage": 0},
    )

    [record] = _records(tmp_path)
    assert record["schema"] == "rlinf.performance-event/v1"
    assert record["trace_id"] == "qualification-a:step-2"
    assert record["timestamp_ns"] == 123
    assert record["request_id"] == 7
    assert record["clock"] == "CLOCK_MONOTONIC"
    assert record["metadata"] == {"pipeline_stage": 0}


def test_record_span_closes_with_error_status(monkeypatch, tmp_path):
    monkeypatch.setenv("RLINF_PERF_MEASUREMENT_DIR", str(tmp_path))
    monkeypatch.setenv("RLINF_PERF_RUN_ID", "qualification-b")

    with pytest.raises(RuntimeError, match="boom"):
        with record_span(
            owner="actor",
            rank=1,
            outer_step=4,
            stage="trainer.update",
        ):
            raise RuntimeError("boom")

    records = _records(tmp_path)
    assert [record["event"] for record in records] == ["start", "end"]
    assert records[0]["status"] == "ok"
    assert records[1]["status"] == "error"
    assert records[0]["timestamp_ns"] <= records[1]["timestamp_ns"]
