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

from pathlib import Path
from types import SimpleNamespace

from toolkits.horde.gr00t_trocar.offline_asset_mirror import (
    install_offline_asset_mirror,
    resolve_mirror_path,
)


def test_resolve_mirror_path_maps_remote_url(tmp_path: Path) -> None:
    asset = tmp_path / "Isaac" / "Healthcare" / "model.usd"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"usd")

    resolved = resolve_mirror_path(
        "https://example.test/Assets/Isaac/Healthcare/model.usd", tmp_path
    )

    assert resolved == asset
    assert resolve_mirror_path("relative/model.usd", tmp_path) is None
    assert resolve_mirror_path("https://example.test/Assets/../escape.usd", tmp_path) is None
    assert resolve_mirror_path("https://example.test/Other/model.usd", tmp_path) is None


def test_install_prefers_mirror_and_preserves_fallback(tmp_path: Path) -> None:
    asset = tmp_path / "model.usd"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"local-mirror")
    calls = []
    module = SimpleNamespace(
        check_file_path=lambda path: calls.append(("check", path)) or 0,
        retrieve_file_path=lambda path, **kwargs: calls.append(("retrieve", path, kwargs))
        or "fallback",
        read_file=lambda path: calls.append(("read", path)) or None,
    )
    url = "https://example.test/Assets/model.usd"

    originals = install_offline_asset_mirror(tmp_path, assets_module=module)

    assert module.check_file_path(url) == 2
    assert module.retrieve_file_path(url) == str(asset)
    assert module.read_file(url).read() == b"local-mirror"
    assert module.check_file_path("missing") == 0
    assert module.retrieve_file_path("missing") == "fallback"
    assert calls == [
        ("check", "missing"),
        ("retrieve", "missing", {"download_dir": None, "force_download": False}),
    ]
    assert set(originals) == {"check_file_path", "retrieve_file_path", "read_file"}


def test_install_rejects_missing_root(tmp_path: Path) -> None:
    module = SimpleNamespace(check_file_path=None, retrieve_file_path=None, read_file=None)
    missing = tmp_path / "missing"

    try:
        install_offline_asset_mirror(missing, assets_module=module)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing mirror root was accepted")
