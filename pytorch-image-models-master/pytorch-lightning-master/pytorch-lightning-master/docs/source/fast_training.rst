.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer

.. _fast_training:

Fast Training
=============
There are multiple options to speed up different parts of the training by choosing to train
on a subset of data. This could be done for speed or debugging purposes.

----------------

Check validation every n epochs
-------------------------------
If you have a small dataset you might want to check validation every n epochs

.. testcode::

    # DEFAULT
    trainer = Trainer(check_val_every_n_epoch=1)

----------------

Force training for min or max epochs
------------------------------------
It can be useful to force training for a minimum number of epochs or limit to a max number.

.. seealso::
    :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    # DEFAULT
    trainer = Trainer(min_epochs=1, max_epochs=1000)

----------------

Set validation check frequency within 1 training epoch
------------------------------------------------------
For large datasets it's often desirable to check validation multiple times within a training loop.
Pass in a float to check that often within 1 training epoch. Pass in an int `k` to check every `k` training batches.
Must use an `int` if using an `IterableDataset`.

.. testcode::

    # DEFAULT
    trainer = Trainer(val_check_interval=0.95)

    # check every .25 of an epoch
    trainer = Trainer(val_check_interval=0.25)

    # check every 100 train batches (ie: for `IterableDatasets` or fixed frequency)
    trainer = Trainer(val_check_interval=100)

----------------

Use data subset for training, validation, and test
--------------------------------------------------
If you don't want to check 100% of the training/validation/test set (for debugging or if it's huge), set these flags.

.. testcode::

    # DEFAULT
    trainer = Trainer(
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0
    )

    # check 10%, 20%, 30% only, respectively for training, validation and test set
    trainer = Trainer(
        limit_train_batches=0.1,
        limit_val_batches=0.2,
        limit_test_batches=0.3
    )

If you also pass ``shuffle=True`` to the dataloader, a different random subset of your dataset will be used for each epoch; otherwise the same subset will be used for all epochs.

.. note:: ``limit_train_batches``, ``limit_val_batches`` and ``limit_test_batches`` will be overwritten by ``overfit_batches`` if ``overfit_batches`` > 0. ``limit_val_batches`` will be ignored if ``fast_dev_run=True``.

.. note:: If you set ``limit_val_batches=0``, validation will be disabled.
