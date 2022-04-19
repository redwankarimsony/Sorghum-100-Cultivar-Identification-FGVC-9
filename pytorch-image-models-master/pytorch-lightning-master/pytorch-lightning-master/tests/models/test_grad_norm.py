# Copyright The PyTorch Lightning team.
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
import os
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest

from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate
from tests.base.develop_utils import reset_seed


class ModelWithManualGradTracker(EvalModelTemplate):
    def __init__(self, norm_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_grad_norms, self.norm_type = [], float(norm_type)

    # validation spoils logger's metrics with `val_loss` records
    validation_step = None
    val_dataloader = None

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # just return a loss, no log or progress bar meta
        x, y = batch
        loss_val = self.loss(y, self(x.flatten(1, -1)))
        return {'loss': loss_val}

    def on_after_backward(self):
        out, norms = {}, []
        prefix = f'grad_{self.norm_type}_norm_'
        for name, p in self.named_parameters():
            if p.grad is None:
                continue

            # `np.linalg.norm` implementation likely uses fp64 intermediates
            flat = p.grad.data.cpu().numpy().ravel()
            norm = np.linalg.norm(flat, self.norm_type)
            norms.append(norm)

            out[prefix + name] = round(norm, 4)

        # handle total norm
        norm = np.linalg.norm(norms, self.norm_type)
        out[prefix + 'total'] = round(norm, 4)
        self.stored_grad_norms.append(out)


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.parametrize("norm_type", [1., 1.25, 2, 3, 5, 10, 'inf'])
def test_grad_tracking(tmpdir, norm_type, rtol=5e-3):
    # rtol=5e-3 respects the 3 decimals rounding in `.grad_norms` and above

    reset_seed()

    # use a custom grad tracking module and a list logger
    model = ModelWithManualGradTracker(norm_type)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        track_grad_norm=norm_type,
        log_every_n_steps=1,  # request grad_norms every batch
    )
    result = trainer.fit(model)

    assert result == 1, "Training failed"
    logged_metrics = trainer.dev_debugger.logged_metrics
    assert len(logged_metrics) == len(model.stored_grad_norms)

    # compare the logged metrics against tracked norms on `.backward`
    for mod, log in zip(model.stored_grad_norms, logged_metrics):
        common = mod.keys() & log.keys()

        log, mod = [log[k] for k in common], [mod[k] for k in common]

        assert np.allclose(log, mod, rtol=rtol)


@pytest.mark.parametrize("log_every_n_steps", [1, 2, 3])
def test_grad_tracking_interval(tmpdir, log_every_n_steps):
    """ Test that gradient norms get tracked in the right interval and that everytime the same keys get logged. """
    trainer = Trainer(
        default_root_dir=tmpdir,
        track_grad_norm=2,
        log_every_n_steps=log_every_n_steps,
        max_steps=10,
    )

    with patch.object(trainer.logger, "log_metrics") as mocked:
        model = EvalModelTemplate()
        trainer.fit(model)
        expected = trainer.global_step // log_every_n_steps
        grad_norm_dicts = []
        for _, kwargs in mocked.call_args_list:
            metrics = kwargs.get("metrics", {})
            grad_norm_dict = {k: v for k, v in metrics.items() if k.startswith("grad_")}
            if grad_norm_dict:
                grad_norm_dicts.append(grad_norm_dict)

        assert len(grad_norm_dicts) == expected
        assert all(grad_norm_dicts[0].keys() == g.keys() for g in grad_norm_dicts)
