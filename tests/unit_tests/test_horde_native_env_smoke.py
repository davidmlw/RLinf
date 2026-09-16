# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from toolkits.horde.gr00t_trocar.native_env_smoke import _write_receipt


def test_write_receipt_atomically_replaces_progress(tmp_path: Path) -> None:
    output = tmp_path / "attempt" / "receipt.json"

    _write_receipt(output, {"status": "pending", "stage": "reset_started"})
    _write_receipt(output, {"status": "passed", "stage": "completed"})

    assert json.loads(output.read_text()) == {
        "stage": "completed",
        "status": "passed",
    }
    assert not output.with_name(f".{output.name}.pending").exists()
