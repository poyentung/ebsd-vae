"""
Indexer for raw (non-latent) diffraction patterns.
"""

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd
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
            with open(self.config.angles_path) as f:
                lines = f.readlines()[2:]  # Skip header lines

            angle_list = []
            for line in lines:
                angles = [angle for angle in line.strip().split(" ") if angle]
                angle_list.append(angles)
            orientations = np.array(angle_list)
            # orientations = (
            #     pd.DataFrame(angle_list, columns=["z1", "x", "z2"])
            #     .astype(float)
            #     .to_numpy()
            # )
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
                num_in_batch = 0  # Initialize num_in_batch

                processed_batch_list = []
                if batch.ndim == 2:  # Single pattern case
                    try:
                        padded_pattern = _pad_pattern_to_size(
                            batch, expected_shape, f"(index {i})"
                        )
                        processed_batch_list.append(padded_pattern)
                        num_in_batch = 1
                    except ValueError as e:
                        # Re-raise with more context if needed, or handle appropriately
                        logger.error(
                            f"Error processing single pattern at index {i}: {e}"
                        )
                        raise
                elif batch.ndim == 3:  # Batch of patterns
                    b = batch.shape[0]
                    num_in_batch = b
                    for pattern_idx in range(b):
                        current_pattern = batch[pattern_idx]
                        try:
                            padded_pattern = _pad_pattern_to_size(
                                current_pattern,
                                expected_shape,
                                f"(batch index {i + pattern_idx})",
                            )
                            processed_batch_list.append(padded_pattern)
                        except ValueError as e:
                            logger.error(
                                f"Error processing pattern in batch at original index {i + pattern_idx}: {e}"
                            )
                            raise

                else:  # Unsupported dimensions
                    raise ValueError(
                        f"Unsupported pattern dimensions ({batch.ndim}) found starting at index {i}"
                    )

                if not processed_batch_list:
                    # Skip if errors occurred and list is empty, or handle as needed
                    progress.update(
                        task, advance=batch.shape[0] if batch.ndim > 1 else 1
                    )  # Advance progress anyway
                    continue

                # Convert list of processed (potentially padded) patterns back to a NumPy array
                processed_batch = np.array(processed_batch_list)

                # Ensure the processed batch has the correct final dimensions before flattening
                if processed_batch.ndim == 3:  # Batch case
                    flattened_batch = processed_batch.reshape(
                        num_in_batch, flattened_dim
                    ).astype(np.float32)
                elif processed_batch.ndim == 2:  # Single pattern case
                    flattened_batch = (
                        processed_batch.flatten().astype(np.float32).reshape(1, -1)
                    )
                else:
                    # This case should ideally not happen if padding/processing is correct
                    raise RuntimeError(
                        f"Unexpected shape after processing batch starting at index {i}: {processed_batch.shape}"
                    )

                flattened_patterns_list.append(flattened_batch)
                progress.update(
                    task, advance=num_in_batch
                )  # Use calculated num_in_batch

        all_flattened_patterns = np.concatenate(flattened_patterns_list, axis=0)

        logger.info(
            f"Adding {len(all_flattened_patterns)} flattened patterns to FAISS database"
        )
        self.db.add_vectors(all_flattened_patterns, orientations)
        logger.info("Saving FAISS database with raw patterns.")
        self.db.save()

    def prepare_pattern(self, pattern: NDArray) -> NDArray[np.float32]:
        """Prepare (validate, pad, and flatten) a single diffraction pattern.

        Args:
            pattern: Diffraction pattern (H, W) as NumPy array.

        Returns:
            Flattened pattern as a float32 numpy array (1, H*W).

        Raises:
            ValueError: If pattern dimensions are incorrect or larger than expected size.
        """
        try:
            padded_pattern = _pad_pattern_to_size(pattern, self.config.image_size)
        except ValueError as e:
            # Re-raise or handle error appropriately
            logger.error(f"Error preparing single pattern: {e}")
            raise

        # Flatten and ensure float32
        flattened_pattern = padded_pattern.flatten().astype(np.float32)
        return flattened_pattern.reshape(1, -1)  # Return as (1, D) for FAISS

    def prepare_patterns_batch(self, patterns: NDArray) -> NDArray[np.float32]:
        """Prepare (validate, pad, and flatten) multiple diffraction patterns.

        Args:
            patterns: Batch of patterns (B, H, W) as NumPy array.

        Returns:
            Flattened patterns as a float32 numpy array (B, H*W).

        Raises:
            ValueError: If patterns dimensions are incorrect or larger than expected size.
        """
        if patterns.ndim != 3:
            raise ValueError(
                f"Input patterns must be 3D (B, H, W), got {patterns.ndim}D"
            )

        batch_size = patterns.shape[0]
        processed_patterns_list = []

        for i in range(batch_size):
            try:
                # Use helper function for each pattern in the batch
                padded_pattern = _pad_pattern_to_size(
                    patterns[i], self.config.image_size, f"(batch index {i})"
                )
                processed_patterns_list.append(padded_pattern)
            except ValueError as e:
                logger.error(f"Error preparing pattern in batch at index {i}: {e}")
                raise  # Re-raise the error to stop processing

        if not processed_patterns_list:
            # This should only happen if the input batch was empty or all patterns failed
            if batch_size > 0:
                raise RuntimeError("Failed to process any patterns in the batch.")
            else:
                # Handle empty input batch if necessary, e.g., return empty array
                return np.empty((0, self.config.dimension), dtype=np.float32)

        # Stack the processed patterns back into a single array
        processed_patterns = np.array(processed_patterns_list)

        # Check if the shape after padding is as expected (B, H, W)
        if processed_patterns.shape != (batch_size, *self.config.image_size):
            # This indicates an issue, maybe an error wasn't caught or logic is flawed
            raise RuntimeError(
                f"Unexpected batch shape after padding: {processed_patterns.shape}. Expected: {(batch_size, *self.config.image_size)}"
            )

        # Flatten the processed (padded) batch
        flattened_dim = self.config.dimension
        flattened_patterns = processed_patterns.reshape(
            batch_size, flattened_dim
        ).astype(np.float32)
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


def _pad_pattern_to_size(
    pattern: NDArray, target_shape: tuple[int, int], pattern_index_info: str = ""
) -> NDArray:
    """Pads a 2D pattern with zeros to match the target shape.

    Args:
        pattern: The 2D input pattern (H, W).
        target_shape: The target (height, width) tuple.
        pattern_index_info: Optional string describing the pattern's origin (e.g., index) for logging.

    Returns:
        The potentially padded pattern.

    Raises:
        ValueError: If the input pattern is larger than the target shape.
    """
    if pattern.ndim != 2:
        raise ValueError(
            f"Padding function requires a 2D pattern, got {pattern.ndim}D {pattern_index_info}"
        )

    target_h, target_w = target_shape
    h, w = pattern.shape

    if h > target_h or w > target_w:
        raise ValueError(
            f"Input pattern shape {pattern.shape} {pattern_index_info} is larger than the target shape {target_shape}."
        )

    if h < target_h or w < target_w:
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded_pattern = np.pad(
            pattern,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        logger.debug(
            f"Padded pattern {pattern_index_info} from {pattern.shape} to {padded_pattern.shape}"
        )
        return padded_pattern.astype(pattern.dtype)  # Ensure dtype consistency
    else:
        # No padding needed
        return pattern
