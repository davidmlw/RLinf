# Copyright 2026 The RLinf Authors.
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

import pytest
import torch

import rlinf.algorithms.registry as algorithm_registry
from rlinf.algorithms.utils import postprocess_loss_metric
from rlinf.utils.metric_utils import materialize_mean_metrics


def test_postprocess_loss_metric_preserves_eager_default():
    metric = torch.tensor(3.25, requires_grad=True)

    result = postprocess_loss_metric({"loss": metric})

    assert result == {"loss": 3.25}


def test_postprocess_loss_metric_can_defer_without_retaining_graph():
    metric = torch.tensor(3.25, requires_grad=True)

    result = postprocess_loss_metric({"loss": metric}, materialize=False)

    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() == 3.25
    assert not result["loss"].requires_grad
    assert result["loss"].grad_fn is None


def test_policy_loss_does_not_forward_materialize_control(monkeypatch):
    received_kwargs = {}

    def fake_loss(**kwargs):
        received_kwargs.update(kwargs)
        return torch.tensor(1.0), {"actor/loss": torch.tensor(2.0, requires_grad=True)}

    monkeypatch.setattr(algorithm_registry, "get_policy_loss", lambda _: fake_loss)
    monkeypatch.setattr(
        algorithm_registry, "preprocess_loss_inputs", lambda **kwargs: kwargs
    )

    _, metrics = algorithm_registry.policy_loss(
        loss_type="test",
        task_type="embodied",
        materialize_metrics=False,
    )

    assert "materialize_metrics" not in received_kwargs
    assert isinstance(metrics["actor/loss"], torch.Tensor)
    assert not metrics["actor/loss"].requires_grad


def test_materialize_mean_metrics_matches_eager_mean():
    metrics = {
        "actor/loss": [
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(2.0),
            torch.tensor(6.0),
        ],
        "actor/lr": [1.0e-4, 2.0e-4, 3.0e-4],
        "actor/mixed": [torch.tensor(2.0), 4.0],
    }

    result = materialize_mean_metrics(metrics)

    assert result["actor/loss"] == pytest.approx(3.0)
    assert result["actor/lr"] == pytest.approx(2.0e-4)
    assert result["actor/mixed"] == pytest.approx(3.0)


def test_materialize_mean_metrics_rejects_non_scalar_tensor():
    with pytest.raises(ValueError, match="must contain scalars"):
        materialize_mean_metrics({"actor/loss": [torch.ones(2)]})


def test_materialize_mean_metrics_rejects_empty_metric():
    with pytest.raises(ValueError, match="has no values"):
        materialize_mean_metrics({"actor/loss": []})
