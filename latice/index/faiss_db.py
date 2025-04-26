"""
Vector database using FAISS for storing and querying latent vectors.

This is a modified version of the chroma_db.py file, with the following changes:
    - Only Flat (exact, brute-force) cosine similarity search is supported.
    - The index is saved and loaded using a single .npz file.
    - The orientation matching query is identical to the one in chroma_db.py.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import faiss
import numpy as np
from numpy.typing import NDArray
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from scipy.spatial.transform import Rotation as R

from latice.index.latent_vector_db_base import (
    LatentVectorDatabaseBase,
    OrientationResult,
)
from latice.utils.utils import QUAT_SYM


logger = logging.getLogger(__name__)


@dataclass
class FaissLatentVectorDatabaseConfig:
    """Configuration for FaissLatentVectorDatabase."""

    npz_path: str = "faiss_index.npz"
    dimension: int = 16


class FaissLatentVectorDatabase(LatentVectorDatabaseBase):
    """Vector database using FAISS for storing and querying latent vectors.

    This class provides methods to create, populate, and query a FAISS
    index containing latent vectors, managing associated orientations separately.
    """

    _orientations: list[NDArray[np.float64]] = field(default_factory=list)
    QUAT_SYM: ClassVar[R] = QUAT_SYM

    def _l2_normalize(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """L2-normalize a batch of vectors along axis 1 (row-wise)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def __init__(self, config: FaissLatentVectorDatabaseConfig | None = None) -> None:
        """Initialise the FAISS latent vector database."""

        self.config = (
            config if config is not None else FaissLatentVectorDatabaseConfig()
        )
        self.dimension = self.config.dimension
        self.base_path = Path(self.config.npz_path)
        self.index_path = self.base_path.with_suffix(".index")
        self.orientations_path = self.base_path.with_suffix(".orient.npz")

        self._orientations = []
        self.index = None

        if self.index_path.exists() and self.orientations_path.exists():
            self.load()
        else:
            logger.info(
                f"Index file ({self.index_path}) or orientations file ({self.orientations_path}) not found. "
                "Creating a new index."
            )
            self.base_path.parent.mkdir(parents=True, exist_ok=True)
            self.index = faiss.index_factory(
                self.dimension,
                "Flat",
                faiss.METRIC_INNER_PRODUCT,  # Always use cosine similarity
            )

    def _validate_vectors(
        self, latent_vectors: NDArray[np.float32], orientations: NDArray[np.float64]
    ) -> None:
        """Validate input vector dimensions and consistency."""
        if len(latent_vectors) != len(orientations):
            raise ValueError("Number of latent vectors and orientations must match")

        if latent_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected latent vectors of dimension {self.dimension}, "
                f"got {latent_vectors.shape[1]}"
            )
        if orientations.shape[1] != 3:
            raise ValueError(
                f"Expected orientations of shape (n, 3), got {orientations.shape}"
            )
        # FAISS typically works best with float32
        if latent_vectors.dtype != np.float32:
            logger.warning("Input latent_vectors are not float32. Casting to float32.")

    def add_vectors(
        self,
        latent_vectors: NDArray[np.float64] | NDArray[np.float32],
        orientations: NDArray[np.float64],
    ) -> None:
        """Add a batch of latent vectors and their orientations to the database.

        Assumes the input arrays represent a single batch.

        Args:
            latent_vectors: Array of latent vectors for the batch (shape: batch_size x dimension).
                          Must be float32. Will be L2-normalized.
            orientations: Array of orientation vectors for the batch (shape: batch_size x 3).
        """
        if self.index is None:
            logger.error("FAISS index not initialized. Cannot add vectors.")
            raise RuntimeError("FAISS index is None.")

        if latent_vectors.dtype != np.float32:
            # Explicitly require float32 input as normalization and FAISS expect it.
            logger.error(
                f"Input latent_vectors must be float32, got {latent_vectors.dtype}."
                " The caller (e.g., RawDiffractionPatternIndexer) should handle casting."
            )
            raise TypeError("FAISS add_vectors requires float32 input.")

        if np.isnan(latent_vectors).any():
            logger.error("NaN values detected in input latent vectors batch.")
            raise ValueError("NaN values found in latent vectors batch.")

        batch_size = len(latent_vectors)
        if batch_size == 0:
            logger.warning("add_vectors called with empty batch. Skipping.")
            return

        self._validate_vectors(latent_vectors, orientations)

        try:
            latent_vectors_normalized = self._l2_normalize(latent_vectors)

            if np.isnan(latent_vectors_normalized).any():
                logger.error(
                    "NaN values detected after L2 normalization. Check for zero vectors."
                )
                raise ValueError(
                    "NaN values after normalization; possible zero vectors."
                )

            self.index.add(latent_vectors_normalized)
            self._orientations.extend(list(orientations))
            logger.debug(
                f"Successfully added batch of {batch_size} vectors to FAISS."
                f" Index total: {self.get_count()}"
            )
        except Exception as e:
            logger.error(f"Error during FAISS index.add or orientation storage: {e}")
            raise

    def create_from_files(
        self, latent_file_path: Path | str, angles_file_path: Path | str
    ) -> None:
        """Create the vector database from latent and angle files.

        Args:
            latent_file_path: Path to the numpy file containing latent vectors.
            angles_file_path: Path to the numpy file containing orientation angles.
        """
        latent_file_path = Path(latent_file_path)
        angles_file_path = Path(angles_file_path)

        logger.info(f"Loading latent vectors from {latent_file_path}")
        latent_vectors = np.load(latent_file_path).astype(np.float32)
        if latent_vectors.ndim > 2:
            logger.warning(
                f"Latent vectors have unexpected shape: {latent_vectors.shape}. "
                "Flattening to (n, d) where n is the number of samples and d is the dimension."
            )
            latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)

        logger.info(f"Loading orientations from {angles_file_path}")
        orientations = np.load(angles_file_path)

        self.add_vectors(latent_vectors, orientations)
        self.save()

    def query_similar(
        self,
        query_vector: NDArray[np.float64] | NDArray[np.float32],
        n_results: int = 20,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Query the database for vectors similar to the query vector (cosine similarity).

        Args:
            query_vector: Query latent vector (1D array). Should be float32. Will be L2-normalized for cosine similarity.
            n_results: Number of similar vectors to return.

        Returns:
            Tuple containing:
                - distances: Array of distances to the nearest neighbors (inner product, i.e., cosine similarity).
                - indices: Array of indices of the nearest neighbors in the FAISS index.
        """
        if self.get_count() == 0:
            logger.warning("Querying an empty index.")
            return np.array([]), np.array([])
        if self.get_count() < n_results:
            logger.warning(
                f"Requested {n_results} results, but index only contains {self.get_count()} vectors. Returning all."
            )
            n_results = self.get_count()

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query vector of dimension {self.dimension}, got {query_vector.shape[1]}"
            )

        query_vector_f32 = query_vector.astype(np.float32)
        query_vector_f32 = self._l2_normalize(query_vector_f32)

        distances, indices = self.index.search(query_vector_f32, n_results)

        # Return distances and indices for the single query vector.
        # FAISS always returns results as (num_queries, n_results) arrays,
        # so for a single query (shape (1, dim)), we take the first row.
        return distances[0], indices[0]

    def find_best_orientation(
        self,
        query_vector: NDArray[np.float64] | NDArray[np.float32],
        top_n: int = 20,
        orientation_threshold: float = 1.0,
        min_required_matches: int = 1,
        max_iterations: int = 3,
    ) -> OrientationResult:
        """Find the best matching orientation for a query vector using FAISS results.

        Args:
            query_vector: Query latent vector.
            top_n: Number of top matches to consider from FAISS search.
            orientation_threshold: Maximum misorientation angle (degrees) to consider similar.
            min_required_matches: Minimum number of similar orientations required.
            max_iterations: Maximum number of reference orientations to try.

        Returns:
            OrientationResult object containing best fit orientation and related data.
        """
        distances, indices = self.query_similar(query_vector, n_results=top_n)

        if len(indices) == 0:
            logger.warning("No similar vectors found for query.")
            return OrientationResult(
                query_vector=query_vector.squeeze(),
                best_orientation=np.array([np.nan, np.nan, np.nan]),
                candidate_orientations=np.array([]),
                distances=np.array([]),
                mean_orientation=None,
                success=False,
                similar_indices=None,
            )

        candidate_orientations = np.array([self._orientations[i] for i in indices])
        rotations = R.from_euler("zxz", candidate_orientations, degrees=True)

        success = False
        best_orientation = candidate_orientations[0]  # Initial best guess
        similar_indices_in_candidates = None
        mean_orientation = None

        actual_iterations = min(max_iterations, len(rotations))

        for iteration in range(actual_iterations):
            ref_rotation = rotations[iteration]
            delta_rotations = ref_rotation.inv() * rotations
            misorientation_angles_rad = delta_rotations.magnitude()
            misorientation_angles_deg = np.degrees(misorientation_angles_rad)

            similar_indices_in_candidates = np.where(
                misorientation_angles_deg < orientation_threshold
            )[0]

            if len(similar_indices_in_candidates) >= min_required_matches:
                similar_orientations_euler = []
                # Find symmetry equivalent orientations for all similar candidates
                for idx in similar_indices_in_candidates:
                    # Find the closest symmetry equivalent to the *reference* rotation
                    sym_equivalent_euler = self._find_symmetry_equivalent_orientation(
                        ref_rotation, rotations[idx]
                    )
                    similar_orientations_euler.append(sym_equivalent_euler)

                if similar_orientations_euler:
                    mean_rotation = R.from_euler(
                        "zxz", np.array(similar_orientations_euler), degrees=True
                    ).mean()
                    mean_orientation = mean_rotation.as_euler("zxz", degrees=True)
                else:
                    mean_orientation = (
                        None  # Should not happen if len >= min_required_matches
                    )

                success = True
                best_orientation = (
                    mean_orientation
                    if mean_orientation is not None
                    else best_orientation
                )
                break  # Found a good consensus

        if not success:
            logger.warning(
                f"Failed to find consensus orientation after {actual_iterations} iterations. "
                f"Best guess is the closest match: {best_orientation}"
            )
            mean_orientation = None  # Explicitly set mean to None if failed
        return OrientationResult(
            query_vector=query_vector.squeeze().astype(
                np.float64
            ),  # Ensure consistent output type
            best_orientation=best_orientation,
            mean_orientation=mean_orientation,
            candidate_orientations=candidate_orientations,
            distances=distances,
            success=success,
            similar_indices=similar_indices_in_candidates,
        )

    def _find_symmetry_equivalent_orientation(
        self, ref_rotation: R, candidate_rotation: R
    ) -> NDArray[np.float64]:
        """Find the symmetry equivalent orientation closest to the reference.

        Identical to the one in chroma_db.py.

        Args:
            ref_rotation: Reference rotation as a scipy Rotation object.
            candidate_rotation: Candidate rotation as a scipy Rotation object.

        Returns:
            Symmetrically equivalent orientation in ZXZ Euler angles (degrees).
        """
        all_sym_rotations = self.QUAT_SYM * candidate_rotation
        delta_rotations = ref_rotation.inv() * all_sym_rotations
        magnitudes = delta_rotations.magnitude()
        closest_sym_idx = magnitudes.argmin()
        sym_equivalent_rotation = all_sym_rotations[closest_sym_idx]
        return sym_equivalent_rotation.as_euler("zxz", degrees=True)

    def find_best_orientations_batch(
        self,
        query_vectors: NDArray[np.float64] | NDArray[np.float32],
        batch_size: int = 32,
        **kwargs,
    ) -> list[OrientationResult]:
        """Process multiple query vectors in batches using FAISS.

        Args:
            query_vectors: Array of query latent vectors (N x dimension).
            batch_size: Batch size for processing (mostly affects progress bar).
            **kwargs: Additional arguments for find_best_orientation.

        Returns:
            List of OrientationResult objects for each query vector.
        """
        n_vectors = len(query_vectors)
        results = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Finding orientations (FAISS)...", total=n_vectors)

            # FAISS search can be batched efficiently, but orientation processing is per-vector
            # We'll still iterate for clarity and applying the orientation logic
            for i in range(0, n_vectors, batch_size):
                batch_end = min(i + batch_size, n_vectors)
                current_batch_vectors = query_vectors[i:batch_end]

                # Process each vector in the current conceptual batch
                for vector in current_batch_vectors:
                    results.append(self.find_best_orientation(vector, **kwargs))
                    progress.update(task, advance=1)
                    # yield results[-1] # Option to yield results incrementally

        return results

    def get_count(self) -> int:
        """Get the number of vectors in the FAISS index."""
        return self.index.ntotal if self.index else 0

    def save(self) -> None:
        """Save the FAISS index to a file and orientations to a separate .npz file."""
        if not self.index:
            logger.error("Cannot save. Index not initialized.")
            return
        if not self._orientations:
            logger.warning("Saving an index with no orientations added yet.")

        self.base_path.parent.mkdir(parents=True, exist_ok=True)

        index_file = str(self.index_path)
        faiss.write_index(self.index, index_file)
        logger.info(f"Saved FAISS index to {index_file}")

        orientations_file = str(self.orientations_path)
        orientations_array = np.array(self._orientations)
        np.savez_compressed(orientations_file, orientations=orientations_array)
        logger.info(f"Saved orientations to {orientations_file}")

    def load(self) -> None:
        """Load the FAISS index using memory mapping and orientations from a separate .npz file."""
        index_file = str(self.index_path)
        orientations_file = str(self.orientations_path)

        if not self.index_path.exists():
            logger.error(f"Cannot load. Index file {index_file} not found.")
            raise FileNotFoundError(f"Index file missing: {index_file}")
        if not self.orientations_path.exists():
            logger.error(
                f"Cannot load. Orientations file {orientations_file} not found."
            )
            raise FileNotFoundError(f"Orientations file missing: {orientations_file}")

        try:
            self.index = faiss.read_index(index_file, faiss.IO_FLAG_MMAP)
            self.dimension = self.index.d  # Update dimension from loaded index
            logger.info(f"Loaded FAISS index from {index_file} using memory mapping.")
        except Exception as e:
            logger.error(f"Error loading FAISS index from {index_file}: {e}")
            raise

        try:
            data = np.load(orientations_file, allow_pickle=True)
            self._orientations = data["orientations"].tolist()
            logger.info(f"Loaded orientations from {orientations_file}")
        except Exception as e:
            logger.error(f"Error loading orientations from {orientations_file}: {e}")
            raise

        if self.index.ntotal != len(self._orientations):
            logger.warning(
                f"Loaded index size ({self.index.ntotal}) does not match number of orientations ({len(self._orientations)})."
                " Files might be out of sync."
            )

    def delete_persistence(self) -> None:
        """Delete the persisted index and orientations files."""
        deleted_files = False

        for file_path in [self.index_path, self.orientations_path]:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted file: {file_path}")
                    deleted_files = True
            except OSError as e:
                logger.error(f"Error deleting file {file_path}: {e}")

        if deleted_files:
            self.index = faiss.index_factory(
                self.dimension,
                "Flat",
                faiss.METRIC_INNER_PRODUCT,  # Always use cosine similarity
            )
            self._orientations = []
