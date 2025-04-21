import math
import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import Optimizer, Adam
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Protocol
from .model import VariationalAutoEncoder
from .utils.utils import plot_detection, log_fig


class OptimizerPartial(Protocol):
    """Callable to instantiate an optimizer."""

    def __call__(self, params: Any) -> Optimizer:
        raise NotImplementedError


class SchedulerPartial(Protocol):
    """Callable to instantiate a learning rate scheduler."""

    def __call__(self, optimizer: Optimizer) -> Any:
        raise NotImplementedError


def get_default_optimiser(params: Any) -> Optimizer:
    """Create the default Adam optimizer."""
    return Adam(params=params, lr=1e-4, weight_decay=0, amsgrad=True)


def get_default_scheduler(optimizer: Optimizer) -> Any:
    """Create the default learning rate scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )


class VAELoss:
    """Loss functions for Variational Autoencoder.

    Encapsulates the different loss functions used in VAE training.
    """

    def __init__(self, kl_lambda: float = 0.1):
        """Initialize VAE loss calculator.

        Args:
            kl_lambda: Weight for the KL divergence term.
        """
        self.kl_lambda = kl_lambda
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(
        self, x_hat: torch.Tensor, logscale: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the Gaussian log-likelihood of the reconstruction.

        Args:
            x_hat: Reconstructed input.
            logscale: Log of the scale parameter.
            x: Original input.

        Returns:
            Gaussian log-likelihood per sample.
        """
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # Measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        # Normalization factor that normalizes the max likelihood to 1
        normalization_factor = torch.log(torch.sqrt(2 * math.pi) * scale)
        log_pxz += normalization_factor

        return log_pxz.mean(dim=(1, 2, 3))

    def binary_cross_entropy(
        self, x_hat: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Calculate binary cross entropy loss.

        Args:
            x_hat: Reconstructed input.
            x: Original input.

        Returns:
            BCE loss per sample.
        """
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(x_hat, x)
        return bce_loss.mean(dim=(1, 2, 3))

    def kl_divergence(
        self, z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Calculate KL divergence between the encoder distribution and prior.

        Uses Monte Carlo estimation of the KL divergence.

        Args:
            z: Sampled latent vector.
            mu: Mean of the encoder distribution.
            std: Standard deviation of the encoder distribution.

        Returns:
            KL divergence per sample.
        """
        # Define the distributions
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # Get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # Sum over last dim to go from single dim distribution to multi-dim
        kl = log_qzx - log_pz
        kl = kl.mean(-1)
        return kl

    def compute_loss(
        self,
        z: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        std: torch.Tensor,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all VAE losses.

        Args:
            z: Sampled latent vector.
            x_hat: Reconstructed input.
            mu: Mean of the encoder distribution.
            std: Standard deviation of the encoder distribution.
            x: Original input.

        Returns:
            Dictionary containing individual loss components and total loss.
        """
        # Reconstruction loss - using BCE
        recon_loss = self.binary_cross_entropy(x_hat, x)

        # KL divergence
        kl = self.kl_divergence(z, mu, std) * self.kl_lambda

        # ELBO (Evidence Lower Bound)
        elbo = kl + recon_loss

        return {
            "loss": elbo.mean(),
            "kl_loss": kl.mean(),
            "recon_loss": recon_loss.mean(),
            "elbo": elbo,  # Per-sample loss for potential analysis
        }


class VAELightningModule(pl.LightningModule):
    """Variational Autoencoder Lightning Module.

    Implements training, validation, and inference procedures for a VAE model.
    Handles optimization, loss calculation, and logging.
    """

    def __init__(
        self,
        model: VariationalAutoEncoder,
        kl_lambda: float = 0.1,
        optimizer_partial: OptimizerPartial = get_default_optimiser,
        lr_scheduler_partial: SchedulerPartial = get_default_scheduler,
    ) -> None:
        """Initialize the VAE Lightning Module.

        Args:
            model: The VAE model architecture.
            kl_lambda: Weight for the KL divergence term in the loss function.
            optimizer_partial: Factory function for creating the optimizer.
            lr_scheduler_partial: Factory function for creating the learning rate scheduler.
        """
        super().__init__()
        self.model = model

        # Loss calculation
        self.loss_fn = VAELoss(kl_lambda=kl_lambda)

        self.optimizer_partial = optimizer_partial
        self.lr_scheduler_partial = lr_scheduler_partial

        # latent representation
        self.latent = []

        # Set random seed for reproducibility
        self._set_random_seeds()

        # Initialize validation and training step outputs
        self.validation_step_outputs = []
        self.training_step_outputs = []  # This was missing

    def _set_random_seeds(self, seed: int = 42) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(x)

    def _get_step_outputs(
        self, batch: tuple[torch.Tensor, torch.Tensor], prefix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Process a batch and compute all outputs and metrics.

        Shared functionality between training and validation steps.

        Args:
            batch: Input batch (x, y).
            prefix: Prefix for metric names (e.g., 'train_', 'val_').

        Returns:
            Dictionary of metrics and outputs.
        """
        x, _ = batch
        z, x_hat, mu, std = self(x)

        # Compute all losses
        losses = self.loss_fn.compute_loss(z, x_hat, mu, std, x)

        # Prepare metrics dictionary with prefixes
        metrics = {
            f"{prefix}loss": losses["loss"],
            f"{prefix}kl_loss": losses["kl_loss"],
            f"{prefix}recon_loss": losses["recon_loss"],
        }

        # Add tensors for visualization if needed
        if prefix == "val_":
            metrics["x"] = x
            metrics["x_hat"] = x_hat

        return metrics

    def training_step(
        self, train_batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Execute a single training step.

        Args:
            train_batch: Batch of training data (x, y).
            batch_idx: Index of the current batch.

        Returns:
            Dictionary of metrics and losses.
        """
        metrics = self._get_step_outputs(train_batch, prefix="train_")

        # Store the output of this training step in self.training_step_outputs
        self.training_step_outputs.append(metrics)  # Save metrics from each batch

        # Log metrics
        self.log("elbo", metrics["train_loss"], prog_bar=True, on_step=True)
        self.log("train_kl_loss", metrics["train_kl_loss"], prog_bar=True, on_step=True)
        self.log("train_recon_loss", metrics["train_recon_loss"], prog_bar=True, on_step=True)

        # Return the loss as required by PyTorch Lightning
        return {"loss": metrics["train_loss"]}

    def on_train_epoch_end(self) -> None:
        """Called at the end of every training epoch."""
        # Aggregate the training step outputs (assuming you want to average them)
        epoch_train_loss = torch.stack([x["train_loss"] for x in self.training_step_outputs]).mean()
        epoch_train_kl_loss = torch.stack([x["train_kl_loss"] for x in self.training_step_outputs]).mean()
        epoch_train_recon_loss = torch.stack([x["train_recon_loss"] for x in self.training_step_outputs]).mean()

        # Log the averaged epoch losses
        self.log("Epoch_train_loss", epoch_train_loss)
        self.log("Epoch_train_kl_loss", epoch_train_kl_loss)
        self.log("Epoch_train_recon_loss", epoch_train_recon_loss)

        # Reset the list of step outputs for the next epoch
        self.training_step_outputs = []  # Clear the outputs for the next epoch

    def validation_step(
        self, val_batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Execute a single validation step."""
        metrics = self._get_step_outputs(val_batch, prefix="val_")

        # Store the output of this validation step in self.validation_step_outputs
        self.validation_step_outputs.append(metrics)  # Save metrics from each batch

        # Log metrics
        self.log("val_loss", metrics["val_loss"], prog_bar=True, on_step=True)
        self.log("val_kl_loss", metrics["val_kl_loss"], prog_bar=True, on_step=True)
        self.log("val_recon_loss", metrics["val_recon_loss"], prog_bar=True, on_step=True)

        return metrics

    def on_validation_epoch_end(self) -> None:
        """Compute the average validation loss over the entire epoch."""
        valid_step_outputs = self.validation_step_outputs  # Now this list is properly populated
        epoch_val_loss = torch.stack([x["val_loss"] for x in valid_step_outputs]).mean()
        epoch_val_kl_loss = torch.stack([x["val_kl_loss"] for x in valid_step_outputs]).mean()
        epoch_val_recon_loss = torch.stack([x["val_recon_loss"] for x in valid_step_outputs]).mean()

        self.log("Epoch_val_loss", epoch_val_loss)
        self.log("Epoch_val_kl_loss", epoch_val_kl_loss)
        self.log("Epoch_val_recon_loss", epoch_val_recon_loss)

        # Example figure logging
        last_batch = (
            valid_step_outputs[-1] if valid_step_outputs[-1]["x"].size(0) >= 4 else valid_step_outputs[-2]
        )
        fig = plot_detection(last_batch["x"], last_batch["x_hat"])
        log_fig(
            fig=fig,
            log_name="reconstruction/eval_check",
            logger=self.logger,
            current_epoch=self.current_epoch,
        )

        # Reset the validation_step_outputs list for the next epoch
        self.validation_step_outputs = []

    def test_step(
        self, test_batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        x, _ = test_batch
        _, _, embeddings, _ = self(x)
        return embeddings

    def test_epoch_end(self, test_step_outputs: list[STEP_OUTPUT]) -> None:
        embeddings = torch.cat([x for x in test_step_outputs], dim=0)
        self.latent = embeddings.detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = self.optimizer_partial(self.model.parameters())
        if self.lr_scheduler_partial:
            scheduler = self.lr_scheduler_partial(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            return optimizer

