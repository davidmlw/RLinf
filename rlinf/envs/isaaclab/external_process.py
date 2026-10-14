# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Optional simulator interpreter behind the existing RLInf environment API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

from .portable_arrays import decode, encode


class ExternalIsaacLabEnv:
    """Keep the simulator resident without changing Ray or model interpreters.

    The CPU IPC copies are an explicit runtime compatibility cost, not a
    performance optimization. RLInf still owns metrics, chunking and PPO.
    """

    def __init__(self, settings: dict[str, Any], *, num_envs: int, seed: int):
        self.timeout = float(settings.get("timeout_seconds", 600))
        root = Path(os.environ["WORKSPACE_ATTEMPT_ROOT"]).resolve()
        runtime = root / "simulators" / uuid.uuid4().hex[:12]
        runtime.mkdir(parents=True)
        ipc_root = Path(os.environ["RLINF_SIDECAR_IPC_ROOT"])
        ipc_root.mkdir(parents=True, exist_ok=True)
        self.ipc = Path(tempfile.mkdtemp(prefix="sim-", dir=ipc_root))
        address = str(self.ipc / "control.sock")
        spec = {**settings, "num_envs": num_envs, "seed": seed, "runtime": str(runtime)}
        spec_path = runtime / "input.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        self.log = (runtime / "simulator.log").open("w")
        env = dict(os.environ)
        for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD"):
            env.pop(name, None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        command = [
            str(settings["python"]),
            "-u",
            str(Path(__file__).with_name("sim6_worker.py")),
            "--socket",
            address,
            "--spec",
            str(spec_path),
        ]
        # Listener.accept has no public timeout; bound its underlying socket
        # before launching the child, and close both on every failed startup.
        self.listener = Listener(address, family="AF_UNIX")
        self.listener._listener._socket.settimeout(self.timeout)
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            self.connection = self.listener.accept()
            self._receive(
                timeout=float(settings.get("startup_timeout_seconds", self.timeout))
            )
        except BaseException:
            self._reap()
            raise
        finally:
            self.listener.close()

    def _receive(self, *, timeout: float | None = None) -> Any:
        if not self.connection.poll(self.timeout if timeout is None else timeout):
            raise TimeoutError(
                f"IsaacLab sidecar timed out; child PID {self.process.pid}"
            )
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(f"IsaacLab sidecar failed: {response['error']}")
        return response["result"]

    def _call(self, method: str, **kwargs: Any) -> Any:
        self.connection.send({"method": method, "kwargs": encode(kwargs)})
        return self._receive()

    def reset(self, seed: int | None = None, env_ids: Any = None) -> Any:
        """Return the native reset result, retaining its original tensors."""
        return decode(self._call("reset", seed=seed, env_ids=env_ids), device="cuda:0")

    def step(self, action: Any) -> Any:
        """Execute exactly one physical step in the persistent simulator."""
        return decode(self._call("step", actions=action), device="cuda:0")

    def device(self) -> str:
        """Expose the Ray worker's one visible CUDA device."""
        return "cuda:0"

    def _reap(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()
        shutil.rmtree(self.ipc, ignore_errors=True)

    def close(self) -> None:
        """Drain the owned child, escalating only if that child fails to exit."""
        try:
            if self.process.poll() is None:
                self._call("close")
                self.process.wait(timeout=45)
            if self.process.returncode != 0:
                raise RuntimeError(
                    f"IsaacLab sidecar exited with code {self.process.returncode}"
                )
        finally:
            self.connection.close()
            self._reap()
