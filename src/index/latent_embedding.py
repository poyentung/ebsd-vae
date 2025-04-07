import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from src.data_module import DPDataModule
from src.index.chroma_db import LatentVectorDatabase, LatentVectorDatabaseConfig
from src.model import VariationalAutoEncoder

logger = logging.getLogger(__name__)


@dataclass
class IndexerConfig:
    """Configuration parameters for the diffraction pattern indexer.

    This dataclass encapsulates all configuration parameters needed for
    pattern indexing, model configuration, and data processing.

    Attributes:
        val_data_ratio: Ratio of validation data to total data
        batch_size: Batch size for model training and inference
        n_cpu: Number of CPU cores to use for data loading
        image_size: Size of input images (height, width)
        inplanes: Number of input planes for the model
        learning_rate: Learning rate for optimizer
        decay: Weight decay for optimizer
        kl_lambda: Weight for KL divergence loss term
        optimizer: Optimizer type ("adam", "sgd", etc.)
        lr_scheduler_kw: Optional learning rate scheduler parameters
        model: VAE model class
        random_seed: Random seed for reproducibility
    """

    batch_size: int = 64
    n_cpu: int = 4
    image_size: Tuple[int, int] = (128, 128)
    inplanes: int = 32
    model: VariationalAutoEncoder
    db_config: LatentVectorDatabaseConfig = LatentVectorDatabaseConfig()


class LatentVectorDataset(Dataset):
    """Dataset class for loading and processing latent vectors."""

    def __init__(self, latent_file_path: Path, device: torch.device) -> None:
        """Initialise the dataset with latent vectors from a numpy file.

        Args:
            latent_file_path: Path to the numpy file containing latent vectors
            device: Device to load the tensors to (CPU, MPS or GPU)
        """
        vectors = np.load(latent_file_path)
        self.vectors = torch.from_numpy(vectors).to(device)

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]


class DiffractionPatternIndexer:
    """Class for indexing electron diffraction patterns using a VAE model.

    This class handles the process of encoding diffraction patterns into
    a latent space and finding the best matching orientations from a
    dictionary of pre-computed patterns.

    Attributes:
        model_path: Path to the trained VAE model
        device: Device to use for computation (CPU or GPU)
        config: Configuration parameters for the model and data processing
    """

    def __init__(
        self, config: Optional[IndexerConfig] = None, device: torch.device | None = None
    ) -> None:
        """Initialize the diffraction pattern indexer.

        Args:
            model_path: Path to the trained VAE model
            device: Device to use for computation; if None, uses GPU if available
            config: Configuration parameters (default configuration used if None)
        """
        self.db = LatentVectorDatabase(config.db_config)
        self.config = config if config is not None else IndexerConfig()

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")

        self.model = self.config.model
        self.model.eval()
        self.model.to(self.device)

    def generate_latent_vectors(
        self, pattern_path: Path, angles_path: Path, output_path: Path | None = None
    ) -> NDArray[np.float64]:
        """Generate latent vectors from diffraction patterns.

        Args:
            pattern_path: Path to the patterns numpy file
            angles_path: Path to the rotation angles file
            output_path: Optional path to save the generated latent vectors

        Returns:
            Normalised latent vectors as a numpy array
        """
        data_module = self._create_data_module(pattern_path, angles_path)
        latent_vectors = self._extract_latent_vectors(self.model, data_module)

        latent_array = np.array(latent_vectors)

        if output_path:
            np.save(output_path, latent_array)
            logger.info(f"Latent vectors saved to {output_path}")

        return latent_array

    def _create_data_module(
        self, pattern_path: Path, rot_angles_path: Path
    ) -> DPDataModule:
        """Create a data module for the given pattern and rotation angles."""
        return DPDataModule(
            path=pattern_path,
            rot_angles_path=rot_angles_path,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
        )

    def _extract_latent_vectors(
        self, model: VariationalAutoEncoder, data_module: DPDataModule
    ) -> list[NDArray[np.float64]]:
        """Extract latent vectors from patterns using the VAE model."""
        latent_vectors = []
        total_patterns = data_module.dataset_full.ebsp_dataset.shape[0]
        logger.info(f"Processing {total_patterns} patterns...")

        with torch.no_grad():
            for i in range(total_patterns):
                data, _ = data_module.dataset_full[i]
                data = data.unsqueeze(0).to(self.device)

                # Encode pattern to latent space
                enc = model.encoder(data)
                mu = model.mu(enc.flatten(1, -1))
                logvar = model.logvar(enc.flatten(1, -1))
                z = model.reparameterize(mu, logvar)
                latent_vectors.append(z.cpu().numpy().squeeze())

        return latent_vectors
