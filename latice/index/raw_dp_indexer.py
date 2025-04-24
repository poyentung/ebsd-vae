"""
Indexer for raw (non-latent) diffraction patterns.
"""

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)
from latice.index.faiss_db import (
    FaissLatentVectorDatabase,
    FaissLatentVectorDatabaseConfig,
)
from latice.index.latent_vector_db_base import OrientationResult

logger = logging.getLogger(__name__)


@dataclass
class RawIndexerConfig:
    """Configuration for the raw diffraction pattern indexer.

    Attributes:
        pattern_path: Path to diffraction patterns numpy file (.npy)
        angles_path: Path to orientation angles numpy file (.npy)
        batch_size: Batch size for processing patterns during queries (optional).
        random_seed: Random seed for reproducibility (used for NumPy only).
        image_size: Size of input diffraction patterns (determines dimension).
        top_n: Number of top matches to consider for orientation finding.
        orientation_threshold: Maximum misorientation angle (degrees) to consider similar.
        db_path: Path for the persisted FAISS database file (.npz).
    """

    pattern_path: Path
    angles_path: Path
    batch_size: int = 64
    random_seed: int = 42
    image_size: tuple[int, int] = (128, 128)
    top_n: int = 20
    orientation_threshold: float = 3.0
    db_path: str = "faiss_raw_index.npz"

    @property
    def dimension(self) -> int:
        """Return the flattened dimension of the image."""
        return self.image_size[0] * self.image_size[1]


class RawDiffractionPatternIndexer:
    """Indexes raw diffraction patterns directly using a vector database.

    This class handles indexing raw patterns without VAE encoding:
    loading patterns/angles from .npy files, flattening patterns,
    storing them with orientations in a vector database,
    and retrieving the best matching orientations for new patterns.
    It uses FaissLatentVectorDatabase configured for high dimensions.
    """

    def __init__(
        self, config: RawIndexerConfig, db: FaissLatentVectorDatabase | None = None
    ) -> None:
        """Initialize the raw indexer.

        Args:
            config: Raw indexer configuration parameters.
            db: Optional pre-configured FaissLatentVectorDatabase. If None, one is created.
        """
        self.config = config
        faiss_config = FaissLatentVectorDatabaseConfig(
            npz_path=self.config.db_path, dimension=self.config.dimension
        )
        self.db = (
            db if db is not None else FaissLatentVectorDatabase(config=faiss_config)
        )

        np.random.seed(self.config.random_seed)

        logger.info(f"Raw index dimensionality: {self.config.dimension}")
        logger.info(f"Using Faiss database with path: {self.db.npz_path}")

    def build_dictionary(self) -> None:
        """Load patterns and angles from .npy files, flatten, and add to the database."""
        logger.info(f"Loading raw patterns from {self.config.pattern_path}")
        try:
            raw_patterns = np.load(self.config.pattern_path)
        except Exception as e:
            logger.error(
                f"Failed to load patterns from {self.config.pattern_path}: {e}"
            )
            raise

        logger.info(f"Loading orientations from {self.config.angles_path}")
        try:
            orientations = np.load(self.config.angles_path)
        except Exception as e:
            logger.error(
                f"Failed to load orientations from {self.config.angles_path}: {e}"
            )
            raise

        if len(raw_patterns) != len(orientations):
            raise ValueError(
                f"Number of patterns ({len(raw_patterns)}) does not match number of orientations ({len(orientations)})."
            )

        logger.info("Validating and flattening patterns...")
        n_patterns = len(raw_patterns)
        flattened_dim = self.config.dimension
        expected_shape = self.config.image_size

        # Prepare patterns in batches to avoid high memory usage if dataset is large
        # Although FAISS add might handle this, pre-flattening ensures validation.
        flattened_patterns_list = []
        batch_size = (
            self.config.batch_size * 10
        )  # Use larger batch for processing numpy arrays

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing patterns...", total=n_patterns)
            for i in range(0, n_patterns, batch_size):
                batch_end = min(i + batch_size, n_patterns)
                batch = raw_patterns[i:batch_end]

                # Basic shape validation (assuming H, W or B, H, W)
                if batch.ndim == 2:  # Single pattern case (less likely for input file)
                    if batch.shape != expected_shape:
                        raise ValueError(
                            f"Pattern at index {i} has shape {batch.shape}, expected {expected_shape}"
                        )
                    flattened_batch = batch.flatten().astype(np.float32).reshape(1, -1)
                elif batch.ndim == 3:  # Batch of patterns
                    if batch.shape[1:] != expected_shape:
                        raise ValueError(
                            f"Patterns starting at index {i} have shape {batch.shape[1:]}, expected {expected_shape}"
                        )
                    num_in_batch = batch.shape[0]
                    flattened_batch = batch.reshape(num_in_batch, flattened_dim).astype(
                        np.float32
                    )
                else:
                    raise ValueError(
                        f"Unsupported pattern dimensions ({batch.ndim}) found starting at index {i}"
                    )

                flattened_patterns_list.append(flattened_batch)
                progress.update(task, advance=len(flattened_batch))

        all_flattened_patterns = np.concatenate(flattened_patterns_list, axis=0)

        logger.info(
            f"Adding {len(all_flattened_patterns)} flattened patterns to FAISS database"
        )
        self.db.add_vectors(all_flattened_patterns, orientations)
        logger.info("Saving FAISS database with raw patterns.")
        self.db.save()

    def prepare_pattern(self, pattern: NDArray) -> NDArray[np.float32]:
        """Prepare (validate and flatten) a single diffraction pattern for indexing.

        Args:
            pattern: Diffraction pattern (H, W) as NumPy array.

        Returns:
            Flattened pattern as a float32 numpy array (1, H*W).
        """
        if pattern.ndim != 2:
            raise ValueError(f"Input pattern must be 2D (H, W), got {pattern.ndim}D")

        if pattern.shape != self.config.image_size:
            raise ValueError(
                f"Input pattern shape {pattern.shape} does not match config image_size {self.config.image_size}."
            )

        # Flatten and ensure float32
        flattened_pattern = pattern.flatten().astype(np.float32)
        return flattened_pattern.reshape(1, -1)  # Return as (1, D) for FAISS

    def prepare_patterns_batch(self, patterns: NDArray) -> NDArray[np.float32]:
        """Prepare (validate and flatten) multiple diffraction patterns for indexing.

        Args:
            patterns: Batch of patterns (B, H, W) as NumPy array.

        Returns:
            Flattened patterns as a float32 numpy array (B, H*W).
        """
        if patterns.ndim != 3:
            raise ValueError(
                f"Input patterns must be 3D (B, H, W), got {patterns.ndim}D"
            )

        if patterns.shape[1:] != self.config.image_size:
            raise ValueError(
                f"Input patterns shape {patterns.shape[1:]} does not match config image_size {self.config.image_size}. Expected (H, W)."
            )

        batch_size = patterns.shape[0]
        flattened_dim = self.config.dimension
        # Flatten each pattern and ensure float32
        flattened_patterns = patterns.reshape(batch_size, flattened_dim).astype(
            np.float32
        )
        return flattened_patterns

    def index_pattern(
        self,
        pattern: NDArray,
        top_n: int | None = None,
        orientation_threshold: float | None = None,
        min_required_matches: int = 18,
        max_iterations: int = 3,
    ) -> OrientationResult:
        """Index a raw diffraction pattern and return the best orientation.

        Args:
            pattern: Raw diffraction pattern to index.
            top_n: Number of top matches to consider.
            orientation_threshold: Maximum misorientation angle to consider.
            min_required_matches: Min matches for orientation consensus.
            max_iterations: Max iterations for orientation consensus.

        Returns:
            Best matching orientation result.
        """
        top_n = top_n or self.config.top_n
        orientation_threshold = (
            orientation_threshold or self.config.orientation_threshold
        )
        query_vector = self.prepare_pattern(pattern)

        orientation_result = self.db.find_best_orientation(
            query_vector,
            top_n=top_n,
            orientation_threshold=orientation_threshold,
            min_required_matches=min_required_matches,
            max_iterations=max_iterations,
        )

        return orientation_result

    def index_patterns_batch(
        self, patterns: NDArray, **kwargs
    ) -> list[OrientationResult]:
        """Index multiple raw diffraction patterns in a batch.

        Args:
            patterns: Batch of raw diffraction patterns (NumPy array, B x H x W).
            **kwargs: Additional arguments for find_best_orientation.

        Returns:
            List of OrientationResult objects for each pattern.
        """
        query_vectors = self.prepare_patterns_batch(patterns)

        orientation_results = self.db.find_best_orientations_batch(
            query_vectors, batch_size=self.config.batch_size, **kwargs
        )
        return orientation_results
