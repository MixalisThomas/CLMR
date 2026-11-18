import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor

from simclr import SimCLR
from simclr.modules import NT_Xent, LARS

from clmr.modules.sgap import SGAPContrastive


class ContrastiveLearning(LightningModule):
    def __init__(self, args, encoder: nn.Module):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder = encoder
        self.n_features = (
            self.encoder.fc.in_features
        )  # get dimensions of last fully-connected layer
        self.model = SimCLR(self.encoder, self.hparams.projection_dim, self.n_features)
        self.criterion = self.configure_criterion()

    def forward(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        _, _, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, _) -> Tensor:
        x, _ = batch
        x_i = x[:, 0, :]
        x_j = x[:, 1, :]
        loss = self.forward(x_i, x_j)
        self.log("Train/loss", loss)
        return loss

    def configure_criterion(self) -> nn.Module:
        # PT lightning aggregates differently in DP mode
        if self.hparams.accelerator == "dp" and self.hparams.gpus:
            batch_size = int(self.hparams.batch_size / self.hparams.gpus)
        else:
            batch_size = self.hparams.batch_size

        # --batch_size is interpreted as PER-DEVICE under DDP, so with multiple
        # GPUs the true contrastive batch is only reconstructed by gathering
        # embeddings across all of them (see GatherLayer usage in NT_Xent /
        # SGAPContrastive's forward). world_size must match the actual number
        # of devices in use for that gather to be correct -- hardcoding 1 here
        # would silently train each GPU on its own local negatives only.
        world_size = self.hparams.gpus if self.hparams.gpus and self.hparams.gpus > 0 else 1

        loss_name = getattr(self.hparams, "loss", "nt_xent")
        if loss_name == "nt_xent":
            criterion = NT_Xent(batch_size, self.hparams.temperature, world_size=world_size)
        elif loss_name == "sgap":
            criterion = SGAPContrastive(batch_size, self.hparams.temperature, world_size=world_size)
        else:
            raise NotImplementedError(f"Unknown loss: {loss_name}")
        return criterion

    def configure_optimizers(self) -> dict:
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * self.hparams.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.hparams.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hparams.max_epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}
