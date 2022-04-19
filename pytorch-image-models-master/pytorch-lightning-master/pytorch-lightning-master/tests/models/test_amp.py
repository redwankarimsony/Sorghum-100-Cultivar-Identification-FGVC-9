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

import pytest
import torch

import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities import APEX_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


@pytest.mark.skip(reason='dp + amp not supported currently')  # TODO
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_amp_single_gpu_dp(tmpdir):
    """Make sure DP/DDP + AMP work."""
    tutils.reset_seed()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=1,
        accelerator='dp',
        precision=16,
    )

    model = EvalModelTemplate()
    # tutils.run_model_test(trainer_options, model)
    result = trainer.fit(model)

    assert result == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_amp_single_gpu_ddp_spawn(tmpdir):
    """Make sure DP/DDP + AMP work."""
    tutils.reset_seed()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=1,
        accelerator='ddp_spawn',
        precision=16,
    )

    model = EvalModelTemplate()
    # tutils.run_model_test(trainer_options, model)
    result = trainer.fit(model)

    assert result == 1


@pytest.mark.skip(reason='dp + amp not supported currently')  # TODO
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_amp_multi_gpu_dp(tmpdir):
    """Make sure DP/DDP + AMP work."""
    tutils.reset_seed()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=2,
        accelerator='dp',
        precision=16,
    )

    model = EvalModelTemplate()
    # tutils.run_model_test(trainer_options, model)
    result = trainer.fit(model)

    assert result == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_amp_multi_gpu_ddp_spawn(tmpdir):
    """Make sure DP/DDP + AMP work."""
    tutils.reset_seed()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=2,
        accelerator='ddp_spawn',
        precision=16,
    )

    model = EvalModelTemplate()
    # tutils.run_model_test(trainer_options, model)
    result = trainer.fit(model)

    assert result == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_amp_gpu_ddp_slurm_managed(tmpdir):
    """Make sure DDP + AMP work."""
    # simulate setting slurm flags
    tutils.set_random_master_port()
    os.environ['SLURM_LOCALID'] = str(0)

    model = EvalModelTemplate()

    # exp file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=[0],
        accelerator='ddp_spawn',
        precision=16,
        callbacks=[checkpoint],
        logger=logger,
    )
    trainer.is_slurm_managing_tasks = True
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test root model address
    assert trainer.slurm_connector.resolve_root_node_address('abc') == 'abc'
    assert trainer.slurm_connector.resolve_root_node_address('abc[23]') == 'abc23'
    assert trainer.slurm_connector.resolve_root_node_address('abc[23-24]') == 'abc23'
    assert trainer.slurm_connector.resolve_root_node_address('abc[23-24, 45-40, 40]') == 'abc23'


@pytest.mark.parametrize("enable_pl_optimizer", [False, True])
def test_cpu_model_with_amp(enable_pl_optimizer, tmpdir):
    """Make sure model trains on CPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
        precision=16,
        enable_pl_optimizer=enable_pl_optimizer,
    )

    model = EvalModelTemplate()

    with pytest.raises((MisconfigurationException, ModuleNotFoundError)):
        tpipes.run_model_test(trainer_options, model, on_gpu=False)


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_amp_without_apex(tmpdir):
    """Check that even with apex amp type without requesting precision=16 the amp backend is void."""
    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        amp_backend='native',
    )
    assert trainer.amp_backend is None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        amp_backend='apex',
    )
    assert trainer.amp_backend is None
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED
    assert trainer.dev_debugger.count_events('AMP') == 0


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(not APEX_AVAILABLE, reason="test requires apex")
def test_amp_with_apex(tmpdir):
    """Check calling apex scaling in training."""

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        precision=16,
        amp_backend='apex',
        gpus=1,
    )
    assert str(trainer.amp_backend) == "AMPType.APEX"
    trainer.fit(model)
    assert trainer.state == TrainerState.FINISHED
    assert trainer.dev_debugger.count_events('AMP') == 10
