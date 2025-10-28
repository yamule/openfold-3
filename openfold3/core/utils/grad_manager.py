# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
import torch
import torch.distributed as dist

from openfold3.core.utils.tensor_utils import tensor_tree_map


class PerSampleGradManager:
    """
    Manages manual optimization for per-sample gradient clipping and accumulation.
    Manual optimization is required because PyTorch Lightning does not natively support
    per-sample gradient clipping, and instead performs this at the batch level.
    """

    def __init__(
        self,
        gradient_clip_val: int | float | None = None,
        accumulate_grad_batches: int = 1,
        log_grad_norm: bool = False,
    ):
        self.max_grad_norm = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_grad_norm = log_grad_norm

        self.grad_accumulator = None

        # Pointers to these objects will be linked in the setup() call
        self._model = None
        self._trainer = None
        self._logger = None
        self._device = None

    def setup(
        self, model: torch.nn.Module, trainer: "pl.Trainer", logger: "pl.loggers.Logger"
    ):
        """
        Initializes the gradient accumulator and links essential components.
        This must be called from the LightningModule's setup() hook.

        Args:
            model: The model instance
            trainer: The pl.Trainer instance
            logger: The logger instance
        """
        self._model = model
        self._trainer = trainer
        self._logger = logger

        self.grad_accumulator = [
            torch.zeros_like(p, requires_grad=False)
            for p in self._model.parameters()
            if p.requires_grad
        ]

        self._device = next(self._model.parameters()).device

    def _clip_grads(self):
        """Clips the gradients currently stored in self._model.parameters()"""

        # Calculate the total norm of all parameter gradients for this single example
        grads = (
            p.grad.detach() for p in self._model.parameters() if p.grad is not None
        )
        global_norm = torch.sqrt(sum([torch.sum(g.float() ** 2) for g in grads]))

        if self._logger is not None and self.log_grad_norm:
            self._logger.log_metrics(
                {"extra_gradients/global_grad_norm": global_norm},
                step=self._trainer.global_step,
            )

        # Clip norm and compute rescale factor
        # Note: We use maximum here to avoid CPU <-> GPU synchronization that can
        # occur with additional conditional `if global_norm > self.max_grad_norm`
        max_norm = torch.tensor(self.max_grad_norm, device=global_norm.device)
        clip_coef = self.max_grad_norm / torch.maximum(global_norm, max_norm)

        # Rescale gradients
        for p in self._model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef.to(p.dtype))

    def _sync_grads(self):
        """
        Averages and syncs the gradients currently in self._model.parameters()
        across all ranks.
        """
        if self._trainer.world_size > 1:
            for p in self._model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def clip_and_accumulate(self):
        """
        Clips the current per-sample gradient in self._model.parameters()
        and adds it to the internal gradient accumulator.

        This should be called after self.manual_backward(loss),
        inside of a self.no_sync() context.
        """
        # Clip the single-sample grads
        self._clip_grads()

        # Manually accumulate clipped grads
        param_iter = (p for p in self._model.parameters() if p.requires_grad)
        for acc_grad, param in zip(self.grad_accumulator, param_iter):
            if param.grad is not None:
                acc_grad.add_(param.grad.detach())

    def sync_grads(self):
        """
        Prepares the gradients for the optimizer step.
        1. Copies the summed grads from the grad accumulator to
           self._model.parameters().
        2. Averages the grads by the number of accumulation steps.
        3. Syncs the averaged grads across all ranks.

        This should be called before opt.step().
        """
        param_iter = (p for p in self._model.parameters() if p.requires_grad)
        for acc_grad, param in zip(self.grad_accumulator, param_iter):
            param.grad = acc_grad.clone()
            if param.grad is not None:
                param.grad.div_(self.accumulate_grad_batches)

        # Sync the locally-averaged gradients
        self._sync_grads()

    def is_step_ready(self, batch_idx: int) -> bool:
        """
        Checks if the optimizer step should be performed or gradients should be
        accumulated for the current step.
        """
        if self._trainer.is_last_batch:
            return True

        is_last_step_of_cycle = (batch_idx + 1) % self.accumulate_grad_batches == 0
        return is_last_step_of_cycle

    def reset_accumulator(self):
        """
        Resets the gradient accumulator to zeros.
        This should be called after opt.step().
        """
        for acc_grad in self.grad_accumulator:
            acc_grad.zero_()

    @property
    def device(self):
        return self._device

    def to(self, device):
        self.grad_accumulator = tensor_tree_map(
            lambda t: t.to(device), self.grad_accumulator
        )
        self._device = device
        return self
