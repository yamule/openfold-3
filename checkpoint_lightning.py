"""
Demonstrates PyTorch Lightning checkpointing with a version buffer and EMA weights.

Steps:
1. Define BoringModelWithVersion with:
   - version_tensor: a registered buffer (saved in state_dict automatically)
   - ema: ExponentialMovingAverage tracking all parameters
2. Train for a few epochs, updating EMA each step and saving per-epoch checkpoints
3. Load each checkpoint and read back the version number and EMA state
"""

import os
from pathlib import Path
from openfold3.core.utils.checkpoint_loading_utils import load_checkpoint
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint

from openfold3.core.utils.exponential_moving_average import ExponentialMovingAverage


CHECKPOINT_DIR = Path("./checkpoints_demo")
EMA_DECAY = 0.999


class BoringModelWithVersion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32, 2)
        # Register version as a buffer so it's saved in state_dict
        # float32 so EMA's decay math (diff *= 1 - decay) can operate on it.
        # Since version_tensor never changes during training, EMA has no effect on it.
        self.register_buffer(
            "version_tensor", torch.tensor([1, 0, 0], dtype=torch.float32)
        )
        self.ema = ExponentialMovingAverage(model=self, decay=EMA_DECAY)

    @property
    def version(self):
        v = self.version_tensor.long().tolist()
        return f"{v[0]}.{v[1]}.{v[2]}"

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.ema.device != self.device:
            self.ema.to(self.device)
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        self.ema.update(self)
        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint["ema"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def make_dataloader():
    x = torch.randn(128, 32)
    y = torch.randint(0, 2, (128,))
    return DataLoader(TensorDataset(x, y), batch_size=32)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = BoringModelWithVersion()
    print(f"Model version before training: {model.version}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="epoch-{epoch:02d}",
        save_top_k=-1,   # keep all checkpoints
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, make_dataloader())

    print(f"\nSaved checkpoints:")
    for f in sorted(os.listdir(CHECKPOINT_DIR)):
        print(f"  {f}")


def load_and_inspect():
    print("\n--- Loading checkpoints and reading version + EMA ---")
    ckpt_files = sorted(
        f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".ckpt")
    )

    for fname in ckpt_files:
        path = CHECKPOINT_DIR / fname

        # --- Method 1: inspect raw checkpoint dict (no model needed) ---
        raw = load_checkpoint(path)
        epoch = raw.get("epoch", "?")
        version_tensor = raw["ema"]["params"]["version_tensor"]
        v = version_tensor.long().tolist()
        version_str = f"{v[0]}.{v[1]}.{v[2]}"

        print(f"\n[{fname}]")
        print(f"  epoch           : {epoch}")
        print(f"  version (raw)   : {version_str}")

        # --- Method 2: load via LightningModule.load_from_checkpoint ---
        # on_load_checkpoint restores self.ema automatically
        model = BoringModelWithVersion.load_from_checkpoint(path)
        print(f"  version (model) : {model.version}")


if __name__ == "__main__":
    train()
    load_and_inspect()
