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
from abc import ABC

import torch


class ValidationEpochEndVariations(ABC):
    """
    Houses all variations of validation_epoch_end steps
    """
    def validation_epoch_end_no_monitor(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        def _mean(res, key):
            # recursive mean for multilevel dicts
            return torch.stack([x[key] if isinstance(x, dict) else _mean(x, key) for x in res]).mean()

        val_acc_mean = _mean(outputs, 'val_acc')

        # alternate between tensor and scalar
        if self.current_epoch % 2 == 0:
            val_acc_mean = val_acc_mean.item()

        metrics_dict = {'val_acc': val_acc_mean}
        results = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return results

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        def _mean(res, key):
            # recursive mean for multilevel dicts
            return torch.stack([x[key] if isinstance(x, dict) else _mean(x, key) for x in res]).mean()

        val_loss_mean = _mean(outputs, 'val_loss')
        val_acc_mean = _mean(outputs, 'val_acc')

        # alternate between tensor and scalar
        if self.current_epoch % 2 == 0:
            val_loss_mean = val_loss_mean.item()
            val_acc_mean = val_acc_mean.item()

        metrics_dict = {'early_stop_on': val_loss_mean, 'val_acc': val_acc_mean}
        results = {'progress_bar': metrics_dict, 'log': metrics_dict}
        return results

    def validation_epoch_end_return_none(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        def _mean(res, key):
            # recursive mean for multilevel dicts
            return torch.stack([x[key] if isinstance(x, dict) else _mean(x, key) for x in res]).mean()

        return None

    def validation_epoch_end__multiple_dataloaders(self, outputs):
        """
        Called at the end of validation to aggregate outputs

        Args:
            outputs: list of individual outputs of each validation step
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        def _mean(res, key):
            return torch.stack([x[key] for x in res]).mean()

        pbar = {}
        logs = {}
        for dl_output_list in outputs:
            output_keys = dl_output_list[0].keys()
            output_keys = [x for x in output_keys if 'val_' in x]
            for key in output_keys:
                metric_out = _mean(dl_output_list, key)
                pbar[key] = metric_out
                logs[key] = metric_out

        results = {
            'val_loss': torch.stack([v for k, v in pbar.items() if k.startswith('val_loss')]).mean(),
            'progress_bar': pbar,
            'log': logs,
        }
        return results
