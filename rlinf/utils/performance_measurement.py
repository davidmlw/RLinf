# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Low-overhead absolute event recording for performance qualification runs."""

import json
import os
import platform
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator


_SCHEMA = "rlinf.performance-event/v1"
_OUTPUT_ENV = "RLINF_PERF_MEASUREMENT_DIR"
_RUN_ID_ENV = "RLINF_PERF_RUN_ID"
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_writer_lock = threading.Lock()


@lru_cache(maxsize=1)
def _boot_id() -> str:
    try:
        return _BOOT_ID_PATH.read_text(encoding="ascii").strip()
    except OSError:
        return "unavailable"


def measurement_enabled() -> bool:
    """Return whether performance event recording is enabled for this process."""

    return bool(os.environ.get(_OUTPUT_ENV))


def _event_path(owner: str, rank: int) -> Path:
    output_dir = Path(os.environ[_OUTPUT_ENV])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_owner = owner.replace("/", "_")
    return output_dir / f"events-{safe_owner}-rank{rank}-pid{os.getpid()}.jsonl"


def record_event(
    *,
    owner: str,
    rank: int,
    outer_step: int,
    stage: str,
    event: str,
    timestamp_ns: int | None = None,
    request_id: int | None = None,
    status: str = "ok",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one monotonic event without introducing device synchronization."""

    if not measurement_enabled():
        return
    run_id = os.environ.get(_RUN_ID_ENV)
    if not run_id:
        raise RuntimeError(f"{_RUN_ID_ENV} is required when {_OUTPUT_ENV} is set")
    record: dict[str, Any] = {
        "schema": _SCHEMA,
        "run_id": run_id,
        "trace_id": f"{run_id}:step-{int(outer_step)}",
        "outer_step": int(outer_step),
        "owner": owner,
        "rank": int(rank),
        "stage": stage,
        "event": event,
        "timestamp_ns": time.monotonic_ns() if timestamp_ns is None else timestamp_ns,
        "clock": "CLOCK_MONOTONIC",
        "boot_id": _boot_id(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "status": status,
    }
    if request_id is not None:
        record["request_id"] = int(request_id)
    if metadata:
        record["metadata"] = metadata
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with _writer_lock:
        with _event_path(owner, rank).open("a", encoding="utf-8") as output:
            output.write(line + "\n")


@contextmanager
def record_span(
    *,
    owner: str,
    rank: int,
    outer_step: int,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record one completed host-monotonic span as paired start/end events."""

    if not measurement_enabled():
        yield
        return
    start_ns = time.monotonic_ns()
    record_event(
        owner=owner,
        rank=rank,
        outer_step=outer_step,
        stage=stage,
        event="start",
        timestamp_ns=start_ns,
        metadata=metadata,
    )
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        record_event(
            owner=owner,
            rank=rank,
            outer_step=outer_step,
            stage=stage,
            event="end",
            status=status,
            metadata=metadata,
        )
