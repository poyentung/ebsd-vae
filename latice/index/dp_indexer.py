from functools import cached_property
from pydantic.dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)

from build.lib.latice.index.faiss_db import FaissLatentVectorDatabase
from latice.data_module import DPDataModule, create_default_transform
from latice.index.chroma_db import ChromaLatentVectorDatabase, OrientationResult
from latice.index.latent_vector_db_base import LatentVectorDatabaseBase
from latice.model import VariationalAutoEncoder

logger = logging.getLogger(__name__)


@dataclass
class IndexerConfig:
    """Configuration for the diffraction pattern indexer.

    Attributes:
        batch_size: Batch size for processing patterns
        device: Device to use for computation ("cuda", "cpu", or "mps")
        latent_dim: Dimension of the latent space
        random_seed: Random seed for reproducibility
        image_size: Size of input diffraction patterns
        top_n: Number of top matches to consider
        orientation_threshold: Maximum misorientation angle to consider
    """

    pattern_path: Path
    angles_path: Path
    batch_size: int = 64
    device: Literal["cuda", "cpu", "mps"] = "cpu"
    latent_dim: int = 16
    random_seed: int = 42
    image_size: tuple[int, int] = (128, 128)
    top_n: int = 20
    orientation_threshold: float = 3.0


class DiffractionPatternIndexer:
    """Indexes diffraction patterns using a VAE model and vector database.

    This class handles the full indexing pipeline: encoding patterns into
    latent space, storing latent vectors with orientations in a vector database,
    and retrieving the best matching orientations for new patterns.
    """

    def __init__(
        self,
        model: VariationalAutoEncoder,
        db: LatentVectorDatabaseBase | None = None,
        config: IndexerConfig | None = None,
    ) -> None:
        """Initialize the indexer with model and database.

        Args:
            model: Trained VAE model for encoding patterns
            db: Vector database for storing and querying latent vectors
            config: Indexer configuration parameters
        """
        self.config = config if config is not None else IndexerConfig()
        self.db = (
            db
            if db is not None
            else FaissLatentVectorDatabase(dimension=self.config.latent_dim)
        )

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        self.device = torch.device(self.config.device)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def build_dictionary(self) -> None:
        """Generate latent vectors from patterns and add to database.

        Args:
            pattern_path: Path to diffraction patterns
            angles_path: Path to orientation angles
            save_latent: Whether to save latent vectors to disk
            latent_output_path: Path to save latent vectors
        """
        data_module = self._create_dataloader

        logger.info(
            f"Generating latent vectors from patterns in {self.config.pattern_path}"
        )
        latent_vectors, orientations = self._extract_latent_vectors_with_angles(
            data_module
        )

        logger.info(f"Adding {len(latent_vectors)} vectors to database")
        self.db.add_vectors(latent_vectors, orientations)

    def encode_pattern(
        self, pattern: NDArray[np.float64] | torch.Tensor
    ) -> NDArray[np.float64]:
        """Encode a single diffraction pattern to latent space.

        Args:
            pattern: Diffraction pattern to encode

        Returns:
            Latent vector representation
        """
        transform = create_default_transform(self.config.image_size)
        if isinstance(pattern, np.ndarray):
            pattern = transform(pattern)

        if pattern.dim() == 2:
            pattern = pattern.unsqueeze(0)  # Add batch dimension
        if pattern.dim() == 3:
            pattern = pattern.unsqueeze(0)  # Add channel dimension

        pattern = pattern.to(self.device)

        with torch.no_grad():
            _, _, mu, _ = self.model(pattern)
        return mu.cpu().numpy().squeeze()

    def encode_patterns_batch(
        self, patterns: NDArray[np.float64] | torch.Tensor
    ) -> NDArray[np.float64]:
        """Encode multiple diffraction patterns to latent space.

        Args:
            patterns: Batch of diffraction patterns (shape: [batch, height, width])

        Returns:
            Batch of latent vector representations (shape: [batch, latent_dim])
        """
        transform = create_default_transform(self.config.image_size)

        # Handle numpy arrays by converting to tensor
        if isinstance(patterns, np.ndarray):
            # If single image with shape (H,W)
            if patterns.ndim == 2:
                # Add batch and channel dims
                patterns = transform(patterns).unsqueeze(0)
            # If batch of images with shape (B,H,W)
            elif patterns.ndim == 3:
                # Transform each image separately
                batch_size = patterns.shape[0]
                transformed = [transform(patterns[i]) for i in range(batch_size)]
                patterns = torch.stack(transformed)
                # patterns = patterns.unsqueeze(1)
        else:
            if patterns.dim() == 2:
                patterns = patterns.unsqueeze(0).unsqueeze(0)
            elif patterns.dim() == 3:
                patterns = patterns.unsqueeze(1)

        # Ensure patterns shape is [B, C, H, W]
        assert patterns.dim() == 4, f"Expected 4D tensor, got {patterns.dim()}D"

        patterns = patterns.to(self.device)

        batch_size = self.config.batch_size
        n_patterns = patterns.shape[0]
        latent_vectors = []

        with torch.no_grad():
            for i in range(0, n_patterns, batch_size):
                batch = patterns[i : i + batch_size]
                _, _, mu, _ = self.model(batch)
                latent_vectors.append(mu.cpu().numpy())

        return np.vstack(latent_vectors)

    def index_pattern(
        self,
        pattern: NDArray[np.float64] | torch.Tensor,
        top_n: int | None = None,
        orientation_threshold: float | None = None,
    ) -> OrientationResult:
        """Index a diffraction pattern and return the best orientation.

        Args:
            pattern: Diffraction pattern to index
            top_n: Number of top matches to consider
            orientation_threshold: Maximum misorientation angle to consider

        Returns:
            Best matching orientation in ZXZ Euler angles
        """
        # Use default values from config if not specified
        top_n = top_n or self.config.top_n
        orientation_threshold = (
            orientation_threshold or self.config.orientation_threshold
        )
        latent_vector = self.encode_pattern(pattern)
        orientation = self.db.find_best_orientation(
            latent_vector, top_n=top_n, orientation_threshold=orientation_threshold
        )

        return orientation

    def index_patterns_batch(
        self, patterns: NDArray[np.float64] | torch.Tensor, **kwargs
    ) -> NDArray[np.float64]:
        """Index multiple diffraction patterns in a batch.

        Args:
            patterns: Batch of diffraction patterns
            **kwargs: Additional arguments for index_pattern

        Returns:
            Array of best matching orientations
        """
        latent_vectors = self.encode_patterns_batch(patterns)
        orientations = self.db.find_best_orientations_batch(
            latent_vectors, batch_size=self.config.batch_size, **kwargs
        )
        return orientations

    @cached_property
    def _create_dataloader(self) -> DataLoader:
        """Create a data module for the given pattern and rotation angles.

        Args:
            pattern_path: Path to the patterns file
            rot_angles_path: Path to the rotation angles file

        Returns:
            Configured data module
        """
        datamodule = DPDataModule(
            path=self.config.pattern_path,
            rot_angles_path=self.config.angles_path,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
        )
        datamodule.setup("test")
        return datamodule.test_dataloader()

    def _extract_latent_vectors_with_angles(
        self, data_loader: DataLoader
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract latent vectors and corresponding angles from a data module.

        Args:
            data_loader: DataLoader containing patterns and angles

        Returns:
            Tuple of (latent_vectors, orientations)
        """
        latent_vectors, orientations = [], []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            # Create the main task
            task = progress.add_task(
                "[cyan]Processing patterns...", total=len(data_loader)
            )
            with torch.no_grad():
                for batch in data_loader:
                    data, angles = batch
                    data = data.to(self.device)

                    # Encode pattern to latent space
                    _, _, mu, _ = self.model(data)

                    # Convert to numpy and append
                    latent_vectors.append(mu.cpu().numpy())
                    orientations.append(angles.numpy())

                    # Update progress bar
                    progress.update(task, advance=1)

        # Concatenate all batches
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        orientations = np.concatenate(orientations, axis=0)

        return latent_vectors, orientations
