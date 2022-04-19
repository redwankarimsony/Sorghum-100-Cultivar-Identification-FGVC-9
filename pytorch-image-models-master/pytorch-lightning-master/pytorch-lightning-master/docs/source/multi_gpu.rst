.. testsetup:: *

    import torch
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _multi_gpu:

Multi-GPU training
==================
Lightning supports multiple ways of doing distributed training.

.. raw:: html

    <video width="50%" max-width="400px" controls
    poster="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/trainer_flags/yt_thumbs/thumb_multi_gpus.png"
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/yt/Trainer+flags+4-+multi+node+training_3.mp4"></video>

|

----------

Preparing your code
-------------------
To train on CPU/GPU/TPU without changing your code, we need to build a few good habits :)

Delete .cuda() or .to() calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Delete any calls to .cuda() or .to(device).

.. testcode::

    # before lightning
    def forward(self, x):
        x = x.cuda(0)
        layer_1.cuda(0)
        x_hat = layer_1(x)

    # after lightning
    def forward(self, x):
        x_hat = layer_1(x)

Init tensors using type_as and register_buffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When you need to create a new tensor, use `type_as`.
This will make your code scale to any arbitrary number of GPUs or TPUs with Lightning.

.. testcode::

    # before lightning
    def forward(self, x):
        z = torch.Tensor(2, 3)
        z = z.cuda(0)

    # with lightning
    def forward(self, x):
        z = torch.Tensor(2, 3)
        z = z.type_as(x)

The :class:`~pytorch_lightning.core.lightning.LightningModule` knows what device it is on. You can access the reference via `self.device`.
Sometimes it is necessary to store tensors as module attributes. However, if they are not parameters they will
remain on the CPU even if the module gets moved to a new device. To prevent that and remain device agnostic,
register the tensor as a buffer in your modules's `__init__` method with :meth:`~torch.nn.Module.register_buffer`.

.. testcode::

    class LitModel(LightningModule):

        def __init__(self):
            ...
            self.register_buffer("sigma", torch.eye(3))
            # you can now access self.sigma anywhere in your module


Remove samplers
^^^^^^^^^^^^^^^
In PyTorch, you must use `torch.nn.DistributedSampler` for multi-node or TPU training. The
sampler makes sure each GPU sees the appropriate part of your data.

.. testcode::

    # without lightning
    def train_dataloader(self):
        dataset = MNIST(...)
        sampler = None

        if self.on_tpu:
            sampler = DistributedSampler(dataset)

        return DataLoader(dataset, sampler=sampler)

Lightning adds the correct samplers when needed, so no need to explicitly add samplers.

.. testcode::

    # with lightning
    def train_dataloader(self):
        dataset = MNIST(...)
        return DataLoader(dataset)

.. note:: You can disable this behavior with `Trainer(replace_sampler_ddp=False)`

.. note:: For iterable datasets, we don't do this automatically.


Synchronize validation and test logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running in distributed mode, we have to ensure that the validation and test step logging calls are synchronized across processes.
This is done by adding `sync_dist=True` to all `self.log` calls in the validation and test step.
This ensures that each GPU worker has the same behaviour when tracking model checkpoints, which is important for later downstream tasks such as testing the best checkpoint across all workers.

Note if you use any built in metrics or custom metrics that use the :ref:`Metrics API <metrics>`, these do not need to be updated and are automatically handled for you.

.. testcode::

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        # Add sync_dist=True to sync logging across all GPU workers
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        # Add sync_dist=True to sync logging across all GPU workers
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True)


Make models pickleable
^^^^^^^^^^^^^^^^^^^^^^
It's very likely your code is already `pickleable <https://docs.python.org/3/library/pickle.html>`_,
in that case no change in necessary.
However, if you run a distributed model and get the following error:

.. code-block::

    self._launch(process_obj)
    File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 47,
    in _launch reduction.dump(process_obj, fp)
    File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
    _pickle.PicklingError: Can't pickle <function <lambda> at 0x2b599e088ae8>:
    attribute lookup <lambda> on __main__ failed

This means something in your model definition, transforms, optimizer, dataloader or callbacks cannot be pickled, and the following code will fail:

.. code-block:: python

    import pickle
    pickle.dump(some_object)

This is a limitation of using multiple processes for distributed training within PyTorch.
To fix this issue, find your piece of code that cannot be pickled. The end of the stacktrace
is usually helpful.
ie: in the stacktrace example here, there seems to be a lambda function somewhere in the code
which cannot be pickled.

.. code-block::

    self._launch(process_obj)
    File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 47,
    in _launch reduction.dump(process_obj, fp)
    File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
    _pickle.PicklingError: Can't pickle [THIS IS THE THING TO FIND AND DELETE]:
    attribute lookup <lambda> on __main__ failed

----------

Select GPU devices
------------------

You can select the GPU devices using ranges, a list of indices or a string containing
a comma separated list of GPU ids:

.. testsetup::

    k = 1

.. testcode::
    :skipif: torch.cuda.device_count() < 2

    # DEFAULT (int) specifies how many GPUs to use per node
    Trainer(gpus=k)

    # Above is equivalent to
    Trainer(gpus=list(range(k)))

    # Specify which GPUs to use (don't use when running on cluster)
    Trainer(gpus=[0, 1])

    # Equivalent using a string
    Trainer(gpus='0, 1')

    # To use all available GPUs put -1 or '-1'
    # equivalent to list(range(torch.cuda.device_count()))
    Trainer(gpus=-1)

The table below lists examples of possible input formats and how they are interpreted by Lightning.
Note in particular the difference between `gpus=0`, `gpus=[0]` and `gpus="0"`.

+---------------+-----------+---------------------+---------------------------------+
| `gpus`        | Type      | Parsed              | Meaning                         |
+===============+===========+=====================+=================================+
| None          | NoneType  | None                | CPU                             |
+---------------+-----------+---------------------+---------------------------------+
| 0             | int       | None                | CPU                             |
+---------------+-----------+---------------------+---------------------------------+
| 3             | int       | [0, 1, 2]           | first 3 GPUs                    |
+---------------+-----------+---------------------+---------------------------------+
| -1            | int       | [0, 1, 2, ...]      | all available GPUs              |
+---------------+-----------+---------------------+---------------------------------+
| [0]           | list      | [0]                 | GPU 0                           |
+---------------+-----------+---------------------+---------------------------------+
| [1, 3]        | list      | [1, 3]              | GPUs 1 and 3                    |
+---------------+-----------+---------------------+---------------------------------+
| "0"           | str       | [0]                 | GPU 0                           |
+---------------+-----------+---------------------+---------------------------------+
| "3"           | str       | [3]                 | GPU 3                           |
+---------------+-----------+---------------------+---------------------------------+
| "1, 3"        | str       | [1, 3]              | GPUs 1 and 3                    |
+---------------+-----------+---------------------+---------------------------------+
| "-1"          | str       | [0, 1, 2, ...]      | all available GPUs              |
+---------------+-----------+---------------------+---------------------------------+

.. note::

    When specifying number of gpus as an integer `gpus=k`, setting the trainer flag
    `auto_select_gpus=True` will automatically help you find `k` gpus that are not
    occupied by other processes. This is especially useful when GPUs are configured
    to be in "exclusive mode", such that only one process at a time can access them.
    For more details see the :ref:`Trainer guide <trainer>`.


Remove CUDA flags
^^^^^^^^^^^^^^^^^

CUDA flags make certain GPUs visible to your script.
Lightning sets these for you automatically, there's NO NEED to do this yourself.

.. testcode::

    # lightning will set according to what you give the trainer
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

However, when using a cluster, Lightning will NOT set these flags (and you should not either).
SLURM will set these for you.
For more details see the :ref:`SLURM cluster guide <slurm>`.

----------

Distributed modes
-----------------
Lightning allows multiple ways of training

- Data Parallel (`accelerator='dp'`) (multiple-gpus, 1 machine)
- DistributedDataParallel (`accelerator='ddp'`) (multiple-gpus across many machines (python script based)).
- DistributedDataParallel (`accelerator='ddp_spawn'`) (multiple-gpus across many machines (spawn based)).
- DistributedDataParallel 2 (`accelerator='ddp2'`) (DP in a machine, DDP across machines).
- Horovod (`accelerator='horovod'`) (multi-machine, multi-gpu, configured at runtime)
- TPUs (`tpu_cores=8|x`) (tpu or TPU pod)

.. note::
    If you request multiple GPUs or nodes without setting a mode, DDP will be automatically used.

For a deeper understanding of what Lightning is doing, feel free to read this
`guide <https://medium.com/@_willfalcon/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565>`_.



Data Parallel
^^^^^^^^^^^^^
`DataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel>`_ (DP) splits a batch across k GPUs.
That is, if you have a batch of 32 and use DP with 2 gpus, each GPU will process 16 samples,
after which the root node will aggregate the results.

.. warning:: DP use is discouraged by PyTorch and Lightning. Use DDP which is more stable and at least 3x faster

.. testcode::
    :skipif: torch.cuda.device_count() < 2

    # train on 2 GPUs (using DP mode)
    trainer = Trainer(gpus=2, accelerator='dp')

Distributed Data Parallel
^^^^^^^^^^^^^^^^^^^^^^^^^
`DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#distributeddataparallel>`_ (DDP) works as follows:

1. Each GPU across each node gets its own process.

2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.

3. Each process inits the model.

.. note:: Make sure to set the random seed before the instantiation of a ``Trainer()`` so that each model initializes with the same weights.

4. Each process performs a full forward and backward pass in parallel.

5. The gradients are synced and averaged across all processes.

6. Each process updates its optimizer.

.. code-block:: python

    # train on 8 GPUs (same machine (ie: node))
    trainer = Trainer(gpus=8, accelerator='ddp')

    # train on 32 GPUs (4 nodes)
    trainer = Trainer(gpus=8, accelerator='ddp', num_nodes=4)

This Lightning implementation of DDP calls your script under the hood multiple times with the correct environment
variables:

.. code-block:: bash

    # example for 3 GPUs DDP
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=1 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
    MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=2 LOCAL_RANK=0 python my_file.py --gpus 3 --etc

We use DDP this way because `ddp_spawn` has a few limitations (due to Python and PyTorch):

1. Since `.spawn()` trains the model in subprocesses, the model on the main process does not get updated.
2. Dataloader(num_workers=N), where N is large, bottlenecks training with DDP... ie: it will be VERY slow or won't work at all. This is a PyTorch limitation.
3. Forces everything to be picklable.

There are cases in which it is NOT possible to use DDP. Examples are:

- Jupyter Notebook, Google COLAB, Kaggle, etc.
- You have a nested script without a root package
- Your script needs to invoke both `.fit` and `.test`, or one of them multiple times

In these situations you should use `dp` or `ddp_spawn` instead.

Distributed Data Parallel 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In certain cases, it's advantageous to use all batches on the same machine instead of a subset.
For instance, you might want to compute a NCE loss where it pays to have more negative samples.

In  this case, we can use DDP2 which behaves like DP in a machine and DDP across nodes. DDP2 does the following:

1. Copies a subset of the data to each node.

2. Inits a model on each node.

3. Runs a forward and backward pass using DP.

4. Syncs gradients across nodes.

5. Applies the optimizer updates.

.. code-block:: python

    # train on 32 GPUs (4 nodes)
    trainer = Trainer(gpus=8, accelerator='ddp2', num_nodes=4)

Distributed Data Parallel Spawn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`ddp_spawn` is exactly like `ddp` except that it uses .spawn to start the training processes.

.. warning:: It is STRONGLY recommended to use `DDP` for speed and performance.

.. code-block:: python

    mp.spawn(self.ddp_train, nprocs=self.num_processes, args=(model, ))

If your script does not support being called from the command line (ie: it is nested without a root
project module) you can use the following method:

.. code-block:: python

    # train on 8 GPUs (same machine (ie: node))
    trainer = Trainer(gpus=8, accelerator='ddp')

We STRONGLY discourage this use because it has limitations (due to Python and PyTorch):

1. The model you pass in will not update. Please save a checkpoint and restore from there.
2. Set Dataloader(num_workers=0) or it will bottleneck training.

`ddp` is MUCH faster than `ddp_spawn`. We recommend you

1. Install a top-level module for your project using setup.py

.. code-block:: python

    # setup.py
    #!/usr/bin/env python

    from setuptools import setup, find_packages

    setup(name='src',
          version='0.0.1',
          description='Describe Your Cool Project',
          author='',
          author_email='',
          url='https://github.com/YourSeed',  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
          install_requires=[
                'pytorch-lightning'
          ],
          packages=find_packages()
          )

2. Setup your project like so:

.. code-block:: bash

    /project
        /src
            some_file.py
            /or_a_folder
        setup.py

3. Install as a root-level package

.. code-block:: bash

    cd /project
    pip install -e .

You can then call your scripts anywhere

.. code-block:: bash

    cd /project/src
    python some_file.py --accelerator 'ddp' --gpus 8


Horovod
^^^^^^^
`Horovod <http://horovod.ai>`_ allows the same training script to be used for single-GPU,
multi-GPU, and multi-node training.

Like Distributed Data Parallel, every process in Horovod operates on a single GPU with a fixed
subset of the data.  Gradients are averaged across all GPUs in parallel during the backward pass,
then synchronously applied before beginning the next step.

The number of worker processes is configured by a driver application (`horovodrun` or `mpirun`). In
the training script, Horovod will detect the number of workers from the environment, and automatically
scale the learning rate to compensate for the increased total batch size.

Horovod can be configured in the training script to run with any number of GPUs / processes as follows:

.. code-block:: python

    # train Horovod on GPU (number of GPUs / machines provided on command-line)
    trainer = Trainer(accelerator='horovod', gpus=1)

    # train Horovod on CPU (number of processes / machines provided on command-line)
    trainer = Trainer(accelerator='horovod')

When starting the training job, the driver application will then be used to specify the total
number of worker processes:

.. code-block:: bash

    # run training with 4 GPUs on a single machine
    horovodrun -np 4 python train.py

    # run training with 8 GPUs on two machines (4 GPUs each)
    horovodrun -np 8 -H hostname1:4,hostname2:4 python train.py

See the official `Horovod documentation <https://horovod.readthedocs.io/en/stable>`_ for details
on installation and performance tuning.

DP/DDP2 caveats
^^^^^^^^^^^^^^^
In DP and DDP2 each GPU within a machine sees a portion of a batch.
DP and ddp2 roughly do the following:

.. testcode::

    def distributed_forward(batch, model):
        batch = torch.Tensor(32, 8)
        gpu_0_batch = batch[:8]
        gpu_1_batch = batch[8:16]
        gpu_2_batch = batch[16:24]
        gpu_3_batch = batch[24:]

        y_0 = model_copy_gpu_0(gpu_0_batch)
        y_1 = model_copy_gpu_1(gpu_1_batch)
        y_2 = model_copy_gpu_2(gpu_2_batch)
        y_3 = model_copy_gpu_3(gpu_3_batch)

        return [y_0, y_1, y_2, y_3]

So, when Lightning calls any of the `training_step`, `validation_step`, `test_step`
you will only be operating on one of those pieces.

.. testcode::

    # the batch here is a portion of the FULL batch
    def training_step(self, batch, batch_idx):
        y_0 = batch

For most metrics, this doesn't really matter. However, if you want
to add something to your computational graph (like softmax)
using all batch parts you can use the `training_step_end` step.

.. testcode::

    def training_step_end(self, outputs):
        # only use when  on dp
        outputs = torch.cat(outputs, dim=1)
        softmax = softmax(outputs, dim=1)
        out = softmax.mean()
        return out

In pseudocode, the full sequence is:

.. code-block:: python

    # get data
    batch = next(dataloader)

    # copy model and data to each gpu
    batch_splits = split_batch(batch, num_gpus)
    models = copy_model_to_gpus(model)

    # in parallel, operate on each batch chunk
    all_results = []
    for gpu_num in gpus:
        batch_split = batch_splits[gpu_num]
        gpu_model = models[gpu_num]
        out = gpu_model(batch_split)
        all_results.append(out)

    # use the full batch for something like softmax
    full out = model.training_step_end(all_results)

To illustrate why this is needed, let's look at DataParallel

.. testcode::

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(batch)

        # on dp or ddp2 if we did softmax now it would be wrong
        # because batch is actually a piece of the full batch
        return y_hat

    def training_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch

        # do softmax here
        outputs = torch.cat(outputs, dim=1)
        softmax = softmax(outputs, dim=1)
        out = softmax.mean()

        return out

If `training_step_end` is defined it will be called regardless of TPU, DP, DDP, etc... which means
it will behave the same regardless of the backend.

Validation and test step have the same option when using DP.

.. testcode::

    def validation_step_end(self, batch_parts_outputs):
        ...

    def test_step_end(self, batch_parts_outputs):
        ...


Distributed and 16-bit precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Due to an issue with Apex and DataParallel (PyTorch and NVIDIA issue), Lightning does
not allow 16-bit and DP training. We tried to get this to work, but it's an issue on their end.

Below are the possible configurations we support.

+-------+---------+----+-----+---------+------------------------------------------------------------+
| 1 GPU | 1+ GPUs | DP | DDP | 16-bit  | command                                                    |
+=======+=========+====+=====+=========+============================================================+
| Y     |         |    |     |         | `Trainer(gpus=1)`                                          |
+-------+---------+----+-----+---------+------------------------------------------------------------+
| Y     |         |    |     | Y       | `Trainer(gpus=1, precision=16)`                            |
+-------+---------+----+-----+---------+------------------------------------------------------------+
|       | Y       | Y  |     |         | `Trainer(gpus=k, accelerator='dp')`                        |
+-------+---------+----+-----+---------+------------------------------------------------------------+
|       | Y       |    | Y   |         | `Trainer(gpus=k, accelerator='ddp')`                       |
+-------+---------+----+-----+---------+------------------------------------------------------------+
|       | Y       |    | Y   | Y       | `Trainer(gpus=k, accelerator='ddp', precision=16)`         |
+-------+---------+----+-----+---------+------------------------------------------------------------+


Implement Your Own Distributed (DDP) training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you need your own way to init PyTorch DDP you can override :meth:`pytorch_lightning.plugins.ddp_plugin.DDPPlugin.init_ddp_connection`.

If you also need to use your own DDP implementation, override: :meth:`pytorch_lightning.plugins.ddp_plugin.DDPPlugin.configure_ddp`.


----------

.. _model-parallelism:

Model Parallelism [BETA]
------------------------

Model Parallelism tackles training large models on distributed systems, by modifying distributed communications and memory management of the model.
Unlike data parallelism, the model is partitioned in various ways across the GPUs, in most cases to reduce the memory overhead when training large models.
This is useful when dealing with large Transformer based models, or in environments where GPU memory is limited.

Lightning currently offers the following methods to leverage model parallelism:

- Sharded Training (partitioning your gradients and optimizer state across multiple GPUs, for reduced memory overhead with **no performance loss**)
- Sequential Model Parallelism with Checkpointing (partition your :class:`nn.Sequential <torch.nn.Sequential>` module across multiple GPUs, leverage checkpointing and microbatching for further memory improvements and device utilization)

Sharded Training
^^^^^^^^^^^^^^^^
Lightning integration of optimizer sharded training provided by `FairScale <https://github.com/facebookresearch/fairscale>`_.
The technique can be found within `DeepSpeed ZeRO <https://arxiv.org/abs/1910.02054>`_ and
`ZeRO-2 <https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/>`_,
however the implementation is built from the ground up to be pytorch compatible and standalone.
Sharded Training allows you to maintain GPU scaling efficiency, whilst reducing memory overhead drastically. In short, expect normal linear scaling, and significantly reduced memory usage when training large models.

Sharded Training still utilizes Data Parallel Training under the hood, except optimizer states and gradients are sharded across GPUs.
This means the memory overhead per GPU is lower, as each GPU only has to maintain a partition of your optimizer state and gradients.

The benefits vary by model and parameter sizes, but we've recorded up to a 63% memory reduction per GPU allowing us to double our model sizes. Because of extremely efficient communication,
these benefits in multi-GPU setups are almost free and throughput scales well with multi-node setups.

Below we use the `NeMo Transformer Lightning Language Modeling example <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling>`_ to benchmark the maximum batch size and model size that can be fit on 8 A100 GPUs for DDP vs Sharded Training.
Note that the benefits can still be obtained using 2 or more GPUs, and for even larger batch sizes you can scale to multiple nodes.

**Increase Your Batch Size**

Use Sharded Training to scale your batch size further using the same compute. This will reduce your overall epoch time.

+----------------------+-----------------------+----------------+---------------------+
| Distributed Training | Model Size (Millions) | Max Batch Size | Percentage Gain (%) |
+======================+=======================+================+=====================+
| Native DDP           | 930                   | 32             | -                   |
+----------------------+-----------------------+----------------+---------------------+
| Sharded DDP          | 930                   | **52**         | **48%**             |
+----------------------+-----------------------+----------------+---------------------+

**Increase Your Model Size**

Use Sharded Training to scale your model size further using the same compute.

+----------------------+------------+---------------------------+---------------------+
| Distributed Training | Batch Size | Max Model Size (Millions) | Percentage Gain (%) |
+======================+============+===========================+=====================+
| Native DDP           | 32         | 930                       | -                   |
+----------------------+------------+---------------------------+---------------------+
| Sharded DDP          | 32         | **1404**                  | **41%**             |
+----------------------+------------+---------------------------+---------------------+
| Native DDP           | 8          | 1572                      | -                   |
+----------------------+------------+---------------------------+---------------------+
| Sharded DDP          | 8          | **2872**                  | **59%**             |
+----------------------+------------+---------------------------+---------------------+

It is highly recommended to use Sharded Training in multi-GPU environments where memory is limited, or where training larger models are beneficial (500M+ parameter models).
A technical note: as batch size scales, storing activations for the backwards pass becomes the bottleneck in training. As a result, sharding optimizer state and gradients becomes less impactful.
Work within the future will bring optional sharding to activations and model parameters to reduce memory further, but come with a speed cost.

To use Sharded Training, you need to first install FairScale using the command below.

.. code-block:: bash

    pip install https://github.com/PyTorchLightning/fairscale/archive/pl_1.1.0.zip


.. code-block:: python

    # train using Sharded DDP
    trainer = Trainer(accelerator='ddp', plugins='ddp_sharded')

Sharded Training can work across all DDP variants by adding the additional ``--plugins ddp_sharded`` flag.

Internally we re-initialize your optimizers and shard them across your machines and processes. We handle all communication using PyTorch distributed, so no code changes are required.

----------

.. _sequential-parallelism:

Sequential Model Parallelism with Checkpointing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyTorch Lightning integration for Sequential Model Parallelism using `FairScale <https://github.com/facebookresearch/fairscale>`_.
Sequential Model Parallelism splits a sequential module onto multiple GPUs, reducing peak GPU memory requirements substantially.
We also provide auto-balancing techniques through FairScale, to find optimal balances for the model across GPUs.
In addition, we use Gradient Checkpointing to reduce GPU memory requirements further, and micro-batches to minimizing device under-utilization automatically.

Reference: https://arxiv.org/abs/1811.06965

.. note:: DDPSequentialPlugin is currently supported only for Pytorch 1.6.

To get started, install FairScale through extras using with ``pip install pytorch-lightning["extra"]``

or directly using

.. code-block:: bash

     pip install https://github.com/PyTorchLightning/fairscale/archive/pl_1.1.0.zip

To use Sequential Model Parallelism, you must define a  :class:`nn.Sequential <torch.nn.Sequential>` module that defines the layers you wish to parallelize across GPUs.
This should be kept within the ``sequential_module`` variable within your ``LightningModule`` like below.

.. code-block:: python

    from pytorch_lightning.plugins.ddp_sequential_plugin import DDPSequentialPlugin
    from pytorch_lightning import LightningModule

    class MyModel(LightningModule):
        def __init__(self):
            ...
            self.sequential_module = torch.nn.Sequential(my_layers)

    # Split my module across 4 gpus, one layer each
    model = MyModel()
    plugin = DDPSequentialPlugin(balance=[1, 1, 1, 1])
    trainer = Trainer(accelerator='ddp', gpus=4, plugins=[plugin])
    trainer.fit(model)


We provide a minimal example of Sequential Model Parallelism using a convolutional model training on cifar10, split onto GPUs `here <https://github.com/PyTorchLightning/pytorch-lightning/tree/master/pl_examples/basic_examples/conv_sequential_example.py>`_.
To run the example, you need to install `Bolts <https://github.com/PyTorchLightning/pytorch-lightning-bolts>`_. Install with ``pip install pytorch-lightning-bolts``.

When running the Sequential Model Parallelism example on 2 GPUS we achieve these memory savings.

.. list-table:: GPU Memory Utilization
   :widths: 25 25 50
   :header-rows: 1

   * - GPUS
     - Without Balancing
     - With Balancing
   * - Gpu 0
     - 4436 MB
     - 1554 MB
   * - Gpu 1
     - ~0
     - 994 MB

To run the example with Sequential Model Parallelism:

.. code-block:: bash

    python pl_examples/basic_examples/conv_sequential_example.py --batch_size 1024 --gpus 2 --accelerator ddp --use_ddp_sequential

To run the same example without Sequential Model Parallelism:

.. code-block:: bash

    python pl_examples/basic_examples/conv_sequential_example.py --batch_size 1024 --gpus 1


Batch size
----------
When using distributed training make sure to modify your learning rate according to your effective
batch size.

Let's say you have a batch size of 7 in your dataloader.

.. testcode::

    class LitModel(LightningModule):

        def train_dataloader(self):
            return Dataset(..., batch_size=7)

In (DDP, Horovod) your effective batch size will be 7 * gpus * num_nodes.

.. code-block:: python

    # effective batch size = 7 * 8
    Trainer(gpus=8, accelerator='ddp|horovod')

    # effective batch size = 7 * 8 * 10
    Trainer(gpus=8, num_nodes=10, accelerator='ddp|horovod')


In DDP2, your effective batch size will be 7 * num_nodes.
The reason is that the full batch is visible to all GPUs on the node when using DDP2.

.. code-block:: python

    # effective batch size = 7
    Trainer(gpus=8, accelerator='ddp2')

    # effective batch size = 7 * 10
    Trainer(gpus=8, num_nodes=10, accelerator='ddp2')


.. note:: Huge batch sizes are actually really bad for convergence. Check out:
        `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>`_

----------

TorchElastic
--------------
Lightning supports the use of TorchElastic to enable fault-tolerant and elastic distributed job scheduling. To use it, specify the 'ddp' or 'ddp2' backend and the number of gpus you want to use in the trainer.

.. code-block:: python

    Trainer(gpus=8, accelerator='ddp')


Following the `TorchElastic Quickstart documentation <https://pytorch.org/elastic/latest/quickstart.html>`_, you then need to start a single-node etcd server on one of the hosts:

.. code-block:: bash

    etcd --enable-v2
         --listen-client-urls http://0.0.0.0:2379,http://127.0.0.1:4001
         --advertise-client-urls PUBLIC_HOSTNAME:2379


And then launch the elastic job with:

.. code-block:: bash

    python -m torchelastic.distributed.launch
            --nnodes=MIN_SIZE:MAX_SIZE
            --nproc_per_node=TRAINERS_PER_NODE
            --rdzv_id=JOB_ID
            --rdzv_backend=etcd
            --rdzv_endpoint=ETCD_HOST:ETCD_PORT
            YOUR_LIGHTNING_TRAINING_SCRIPT.py (--arg1 ... train script args...)


See the official `TorchElastic documentation <https://pytorch.org/elastic>`_ for details
on installation and more use cases.

----------

Jupyter Notebooks
-----------------
Unfortunately any `ddp_` is not supported in jupyter notebooks. Please use `dp` for multiple GPUs. This is a known
Jupyter issue. If you feel like taking a stab at adding this support, feel free to submit a PR!

----------

Pickle Errors
--------------
Multi-GPU training sometimes requires your model to be pickled. If you run into an issue with pickling
try the following to figure out the issue

.. code-block:: python

    import pickle

    model = YourModel()
    pickle.dumps(model)

However, if you use `ddp` the pickling requirement is not there and you should be fine. If you use `ddp_spawn` the
pickling requirement remains. This is a limitation of Python.
