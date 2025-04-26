"""Indexer for raw (non-latent) diffraction patterns."""

from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)
from latice.index.chroma_db import (
    ChromaLatentVectorDatabase,
    ChromaLatentVectorDatabaseConfig,
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
        db_type: Type of vector database to use ("faiss" or "chroma").
        persist_directory: Directory to persist the database. Behavior depends on db_type.
                           For FAISS, the .npz file will be saved here.
                           For ChromaDB, this is the persistence directory.
        collection_name: Name for the ChromaDB collection (used if db_type is "chroma").
        batch_size: Batch size for processing patterns during queries (optional).
        random_seed: Random seed for reproducibility (used for NumPy only).
        image_size: Size of input diffraction patterns (determines dimension).
        top_n: Number of top matches to consider for orientation finding.
        orientation_threshold: Maximum misorientation angle (degrees) to consider similar.
    """

    pattern_path: Path
    angles_path: Path
    db_type: Literal["faiss", "chroma"] = "chroma"
    persist_directory: str = ".vector_db"
    collection_name: str = "raw_patterns"
    batch_size: int = 64
    random_seed: int = 42
    image_size: tuple[int, int] = (128, 128)
    top_n: int = 20
    orientation_threshold: float = 3.0

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
    """

    def __init__(
        self,
        config: RawIndexerConfig,
        db: FaissLatentVectorDatabase | ChromaLatentVectorDatabase | None = None,
    ) -> None:
        """Initialize the raw indexer.

        Args:
            config: Raw indexer configuration parameters.
            db: Optional pre-configured database instance. If None, one is created based on config.db_type.
        """
        self.config = config
        np.random.seed(self.config.random_seed)

        if db is not None:
            self.db = db
            logger.info(f"Using provided {type(db).__name__} instance.")
        elif self.config.db_type == "faiss":
            faiss_path = str(
                Path(self.config.persist_directory)
                / f"{self.config.collection_name}_faiss.npz"
            )
            faiss_config = FaissLatentVectorDatabaseConfig(
                npz_path=faiss_path, dimension=self.config.dimension
            )
            self.db = FaissLatentVectorDatabase(config=faiss_config)
            logger.info(f"Initialized FAISS database at {faiss_path}")
        elif self.config.db_type == "chroma":
            chroma_config = ChromaLatentVectorDatabaseConfig(
                persist_directory=self.config.persist_directory,
                collection_name=self.config.collection_name,
                dimension=self.config.dimension,
            )
            self.db = ChromaLatentVectorDatabase(config=chroma_config)
            logger.info(
                f"Initialized ChromaDB with collection '{self.config.collection_name}' in '{self.config.persist_directory}'"
            )
        else:
            raise ValueError(f"Unsupported db_type: {self.config.db_type}")

        logger.info(f"Raw index dimensionality: {self.config.dimension}")

    def _load_patterns(self) -> NDArray[np.float32]:
        """Load patterns from .npy file using memory mapping."""
        logger.info(
            f"Loading raw patterns from {self.config.pattern_path} using memory map"
        )
        try:
            # Load patterns using memory mapping to avoid loading all into RAM
            raw_patterns_mmap = np.load(self.config.pattern_path, mmap_mode="r")
        except Exception as e:
            logger.error(
                f"Failed to load/mmap patterns from {self.config.pattern_path}: {e}"
            )
            raise
        else:
            return raw_patterns_mmap

    def _load_orientations(self) -> NDArray[np.float64]:
        """Load orientations from .npy file using memory mapping."""
        logger.info(f"Loading orientations from {self.config.angles_path}")
        try:
            # Orientations are assumed to be small enough to load into memory
            with open(self.config.angles_path) as f:
                lines = f.readlines()[2:]  # Skip header lines
            angle_list = []
            for line in lines:
                angles = [angle for angle in line.strip().split(" ") if angle]
                angle_list.append(angles)
            orientations = np.array(angle_list, dtype=np.float64)
        except Exception as e:
            logger.error(
                f"Failed to load orientations from {self.config.angles_path}: {e}"
            )
            raise
        else:
            return orientations

    def _validate_and_get_n_patterns(
        self, raw_patterns_mmap: NDArray[np.float32], orientations: NDArray[np.float64]
    ) -> int:
        """Get the number of patterns in the pattern file."""
        n_patterns = raw_patterns_mmap.shape[0]
        if n_patterns != len(orientations):
            raise ValueError(
                f"Number of patterns ({n_patterns}) does not match number of orientations ({len(orientations)})."
            )
        return n_patterns

    def build_dictionary(self) -> None:
        """Load patterns and angles, process in batches, and add to the database using memory mapping."""
        raw_patterns_mmap = self._load_patterns()
        orientations = self._load_orientations()
        n_patterns = self._validate_and_get_n_patterns(raw_patterns_mmap, orientations)

        logger.info("Processing and adding patterns to DB in batches...")
        flattened_dim = self.config.dimension
        expected_shape = self.config.image_size
        batch_size = self.config.batch_size
        n_batches = (n_patterns + batch_size - 1) // batch_size

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing and adding patterns...", total=n_batches
            )
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, n_patterns)

                # Read only the current batch from the memory-mapped file
                # This loads only this chunk into RAM
                try:
                    current_raw_batch = np.array(
                        raw_patterns_mmap[batch_start:batch_end]
                    )
                    current_orientations = orientations[batch_start:batch_end]
                    num_in_batch = current_raw_batch.shape[0]
                except Exception as e:
                    logger.error(
                        f"Error reading batch {batch_idx} from memory map or orientations: {e}"
                    )
                    raise

                processed_batch_list = []
                # Determine expected dimensions based on raw input
                # Handle case where the input file might contain a single 2D pattern flattened
                if current_raw_batch.ndim == 2 and num_in_batch == 1:
                    # Reshape assumes it's already flattened to H*W, needs original H,W
                    # This case is ambiguous without knowing original H, W. Assuming 3D input for batching.
                    logger.warning(
                        f"Batch {batch_idx}: Encountered 2D array in batch processing, assuming it's a single pattern. Requires 3D input (B, H, W) for robust batching."
                    )
                    # Attempt to process as single pattern, might fail if not H, W shape
                    try:
                        pattern_to_process = current_raw_batch.reshape(expected_shape)
                        padded_pattern = _pad_pattern_to_size(
                            pattern_to_process, expected_shape, f"(index {batch_start})"
                        )
                        processed_batch_list.append(padded_pattern)
                    except ValueError as e:
                        logger.error(
                            f"Error reshaping/padding single 2D pattern in batch {batch_idx}: {e}"
                        )
                        raise

                elif (
                    current_raw_batch.ndim == 3
                ):  # Expected case: Batch of patterns (B, H, W)
                    for pattern_idx in range(num_in_batch):
                        current_pattern = current_raw_batch[pattern_idx]
                        try:
                            padded_pattern = _pad_pattern_to_size(
                                current_pattern,
                                expected_shape,
                                f"(batch index {batch_start + pattern_idx})",
                            )
                            processed_batch_list.append(padded_pattern)
                        except ValueError as e:
                            logger.error(
                                f"Error padding pattern in batch at original index {batch_start + pattern_idx}: {e}"
                            )
                            raise
                else:
                    raise ValueError(
                        f"Unsupported pattern dimensions ({current_raw_batch.ndim}) found in batch {batch_idx}"
                    )

                if not processed_batch_list:
                    logger.warning(
                        f"Batch {batch_idx}: No patterns successfully processed, skipping DB add."
                    )
                    progress.update(task, advance=1)
                    continue

                processed_batch = np.array(processed_batch_list)
                flattened_batch = processed_batch.reshape(
                    num_in_batch, flattened_dim
                ).astype(np.float32)

                try:
                    self.db.add_vectors(flattened_batch, current_orientations)
                    logger.debug(
                        f"Added batch {batch_idx} ({num_in_batch} patterns) to DB."
                    )
                except Exception as e:
                    logger.error(
                        f"Error adding batch {batch_idx} to {type(self.db).__name__}: {e}"
                    )
                    raise

                progress.update(task, advance=1)

        logger.info(
            f"Finished adding {n_patterns} patterns to the database."
            f" Final DB count: {self.db.get_count()}."
        )
        if self.config.db_type == "faiss":
            logger.info("Saving the final database.")
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
            logger.error(f"Error preparing single pattern: {e}")
            raise

        flattened_pattern = padded_pattern.flatten().astype(np.float32)
        return flattened_pattern.reshape(1, -1)

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
                padded_pattern = _pad_pattern_to_size(
                    patterns[i], self.config.image_size, f"(batch index {i})"
                )
                processed_patterns_list.append(padded_pattern)
            except ValueError as e:
                logger.error(f"Error preparing pattern in batch at index {i}: {e}")
                raise

        if not processed_patterns_list:
            if batch_size > 0:
                raise RuntimeError("Failed to process any patterns in the batch.")
            else:
                return np.empty((0, self.config.dimension), dtype=np.float32)

        processed_patterns = np.array(processed_patterns_list)

        # Check if the shape after padding is as expected (B, H, W)
        if processed_patterns.shape != (batch_size, *self.config.image_size):
            raise RuntimeError(
                f"Unexpected batch shape after padding: {processed_patterns.shape}. Expected: {(batch_size, *self.config.image_size)}"
            )

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
        return padded_pattern.astype(pattern.dtype)
    else:
        return pattern
