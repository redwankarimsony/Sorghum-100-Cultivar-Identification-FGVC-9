import os
import platform
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.plugins.sharded_plugin import DDPShardedPlugin
from pytorch_lightning.utilities import FAIRSCALE_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base.boring_model import BoringModel


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
def test_ddp_choice_default_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend.ddp_plugin, DDPPlugin)
            raise RuntimeError('finished plugin check')

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=gpus,
        num_processes=num_processes,
        accelerator=ddp_backend,
        callbacks=[CB()],
    )

    with pytest.raises(RuntimeError, match='finished plugin check'):
        trainer.fit(model)


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
def test_ddp_choice_custom_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):
    class MyDDP(DDPPlugin):
        pass

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend.ddp_plugin, MyDDP)
            raise RuntimeError('finished plugin check')

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=gpus,
        num_processes=num_processes,
        accelerator=ddp_backend,
        plugins=[MyDDP()],
        callbacks=[CB()],
    )

    with pytest.raises(RuntimeError, match='finished plugin check'):
        trainer.fit(model)


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed sharded plugin is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_choice_string_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend.ddp_plugin, DDPShardedPlugin)
            raise RuntimeError('finished plugin check')

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=gpus,
        num_processes=num_processes,
        accelerator=ddp_backend,
        plugins='ddp_sharded',
        callbacks=[CB()],
    )

    with pytest.raises(RuntimeError, match='finished plugin check'):
        trainer.fit(model)


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
def test_ddp_invalid_choice_string_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):
    with pytest.raises(MisconfigurationException, match='not a supported lightning custom plugin'):
        Trainer(
            fast_dev_run=True,
            gpus=gpus,
            num_processes=num_processes,
            accelerator=ddp_backend,
            plugins='invalid',
        )


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed sharded plugin is not supported on Windows")
@pytest.mark.skipif(not FAIRSCALE_AVAILABLE, reason="Fairscale is not available")
def test_ddp_invalid_choice_string_and_custom_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):
    """
    Test passing a lightning custom ddp plugin and a default ddp plugin throws an error.
    """

    class MyDDP(DDPPlugin):
        pass

    with pytest.raises(MisconfigurationException, match='you can only use one DDP plugin in plugins'):
        Trainer(
            fast_dev_run=True,
            gpus=gpus,
            num_processes=num_processes,
            accelerator=ddp_backend,
            plugins=['ddp_sharded', MyDDP()],
        )


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize(
    ["ddp_backend", "gpus", "num_processes"],
    [("ddp_cpu", None, None), ("ddp", 2, 0), ("ddp2", 2, 0), ("ddp_spawn", 2, 0)],
)
def test_ddp_choice_custom_ddp_cpu_custom_args(
        tmpdir, ddp_backend, gpus, num_processes
):
    class MyDDP(DDPPlugin):
        pass

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend.ddp_plugin, MyDDP)
            raise RuntimeError('finished plugin check')

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        gpus=gpus,
        num_processes=num_processes,
        accelerator=ddp_backend,
        plugins=[MyDDP(broadcast_buffers=False, find_unused_parameters=True)],
        callbacks=[CB()],
    )

    with pytest.raises(RuntimeError, match='finished plugin check'):
        trainer.fit(model)
