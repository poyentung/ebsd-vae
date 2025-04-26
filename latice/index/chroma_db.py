import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
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
class ChromaLatentVectorDatabaseConfig:
    """Configuration for LatentVectorDatabase.

    Attributes:
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist the database to disk
            (if None, will use in-memory database)
        dimension: Dimension of the latent vectors
        hnsw_ef_construction: HNSW construction parameter (higher -> more accuracy, longer build).
    """

    collection_name: str = "latent_vectors"
    persist_directory: str | None = ".chroma_db"
    dimension: int = 16
    add_batch_size: int = 1000  # Default batch size for adding vectors


class ChromaLatentVectorDatabase(LatentVectorDatabaseBase):
    """Vector database for storing and querying latent vectors and their orientations.

    This class provides methods to create, populate, and query a ChromaDB
    collection containing latent vectors and their corresponding orientations.

    Attributes:
        config: Configuration for the database
        collection_name: Name of the ChromaDB collection
        client: ChromaDB client instance
        collection: ChromaDB collection instance
        dimension: Dimension of the latent vectors
    """

    def __init__(self, config: ChromaLatentVectorDatabaseConfig | None = None) -> None:
        """Initialize the latent vector database.

        Args:
            config: Configuration for the database
        """
        self.config = (
            config if config is not None else ChromaLatentVectorDatabaseConfig()
        )
        self.collection_name = self.config.collection_name
        self.dimension = self.config.dimension
        self.persist_directory = self.config.persist_directory

        # Initialise ChromaDB client
        if self.persist_directory:
            persist_path = Path(self.persist_directory)
            persist_path.mkdir(exist_ok=True, parents=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"Created persistent ChromaDB at {self.persist_directory}")
        else:
            self.client = chromadb.Client()
            logger.info("Created in-memory ChromaDB")

        # Create or get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Retrieved existing collection '{self.collection_name}'")
        except (ValueError, chromadb.errors.InvalidCollectionException):
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"dimension": self.dimension, "hnsw:space": "cosine"},
            )
            logger.info(f"Created new collection '{self.collection_name}'")

    def _validate_dtype(self, latent_vectors: NDArray[np.float32]) -> None:
        """Check for NaN values in input vectors (before potential casting)."""
        if np.isnan(latent_vectors).any():
            logger.error(
                f"NaN values detected in input {latent_vectors.dtype} latent vectors."
            )
            raise ValueError("NaN values found in latent vectors batch.")

    def _validate_shape(
        self, latent_vectors: NDArray[np.float32], orientations: NDArray[np.float64]
    ) -> bool:
        """Validate shapes and batch consistency. Returns False if batch is empty."""
        batch_size = len(latent_vectors)
        if batch_size == 0:
            logger.warning("add_vectors called with empty batch. Skipping.")
            return False

        if batch_size != len(orientations):
            raise ValueError(
                f"Batch size mismatch: {batch_size} vectors vs {len(orientations)} orientations."
            )
        if latent_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Input vector dimension {latent_vectors.shape[1]} does not match DB dimension {self.dimension}."
            )
        if orientations.shape[1] != 3:
            raise ValueError(
                f"Input orientation shape {orientations.shape} is invalid. Expected (batch_size, 3)."
            )
        return True

    def add_vectors(
        self,
        latent_vectors: NDArray[np.float32] | NDArray[np.float64],
        orientations: NDArray[np.float64],
    ) -> None:
        """Add a batch of latent vectors and their orientations to the database.

        Assumes the input arrays represent a single batch.

        Args:
            latent_vectors: Array of latent vectors for the batch (shape: batch_size x dimension).
                          Will be cast to float32. NaN values will raise ValueError.
            orientations: Array of orientation vectors for the batch (shape: batch_size x 3).
        """
        self._validate_dtype(latent_vectors)

        if latent_vectors.dtype != np.float32:
            latent_vectors_f32 = latent_vectors.astype(np.float32)
            if np.isnan(latent_vectors_f32).any():
                logger.error(
                    "NaN values detected after casting float64 vectors to float32."
                )
                raise ValueError("NaN values found after casting latent vectors.")
        else:
            latent_vectors_f32 = latent_vectors

        if not self._validate_shape(latent_vectors_f32, orientations):
            return

        try:
            current_count = self.get_count()
        except Exception as e:
            logger.error(f"Failed to get current count from ChromaDB: {e}")
            raise

        batch_size = len(latent_vectors_f32)
        batch_ids = [f"vec_{j + current_count}" for j in range(batch_size)]
        batch_vectors_list = latent_vectors_f32.tolist()
        batch_orientations_list = orientations.tolist()

        batch_metadata = [
            {
                "orientation_str": ",".join(map(str, orient)),
                "phi1": float(orient[0]),
                "Phi": float(orient[1]),
                "phi2": float(orient[2]),
            }
            for orient in batch_orientations_list
        ]

        try:
            self.collection.add(
                embeddings=batch_vectors_list, metadatas=batch_metadata, ids=batch_ids
            )
            logger.debug(
                f"Successfully added batch of {batch_size} vectors to ChromaDB."
            )
        except Exception as e:
            logger.error(
                f"Error adding batch to ChromaDB collection '{self.collection_name}': {e}"
            )
            raise  # Re-raise error to signal failure

    def create_from_files(self, latent_file_path: Path, angles_file_path: Path) -> None:
        """Create the vector database from latent and angle files using memory mapping.

        Args:
            latent_file_path: Path to the numpy file containing latent vectors (N x D).
            angles_file_path: Path to the numpy file containing orientation angles (N x 3).
        """
        latent_file_path = Path(latent_file_path)
        angles_file_path = Path(angles_file_path)

        logger.info(
            f"Loading latent vectors from {latent_file_path} using memory mapping."
        )
        # Load latent vectors using memory map mode 'r' (read-only)
        # This avoids loading the entire file into RAM.
        try:
            latent_vectors_mmap = np.load(latent_file_path, mmap_mode="r").astype(
                np.float32
            )
        except MemoryError as e:
            logger.error(
                f"MemoryError while trying to memory-map {latent_file_path}. "
                "Ensure the file is not corrupted and the system has enough address space. Error: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load/mmap latent vectors: {e}")
            raise

        logger.info(f"Loading orientations from {angles_file_path}")
        # Orientations are usually much smaller, load normally
        try:
            orientations = np.load(angles_file_path)
        except Exception as e:
            logger.error(f"Failed to load orientations: {e}")
            raise

        n_samples = self._validate_and_get_n_samples(latent_vectors_mmap, orientations)

        # Use the batch size from config for processing the memory-mapped array
        add_batch_size = self.config.add_batch_size
        n_batches = (n_samples + add_batch_size - 1) // add_batch_size

        logger.info(
            f"Adding {n_samples} vectors in {n_batches} batches using memory-mapped input."
        )

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "Adding vectors from mmap file...", total=n_batches
            )
            current_count = self.get_count()

            for i in range(n_batches):
                start_idx = i * add_batch_size
                end_idx = min((i + 1) * add_batch_size, n_samples)

                # Read only the current batch from the memory-mapped file
                # This loads only the necessary chunk into RAM
                batch_vectors = np.array(latent_vectors_mmap[start_idx:end_idx])
                batch_orientations = orientations[start_idx:end_idx]

                # Reuse the validation logic from add_vectors if needed, or simplify
                if batch_vectors.shape[1] != self.dimension:
                    raise ValueError(
                        f"Batch {i}: Expected vectors of dimension {self.dimension}, got {batch_vectors.shape[1]}"
                    )

                batch_ids = [
                    f"vec_{j + current_count}" for j in range(start_idx, end_idx)
                ]
                batch_metadata = [
                    {
                        "orientation_str": ",".join(map(str, orient)),
                        "phi1": float(orient[0]),
                        "Phi": float(orient[1]),
                        "phi2": float(orient[2]),
                    }
                    for orient in batch_orientations
                ]

                # Add the current batch to the collection
                try:
                    self.collection.add(
                        embeddings=batch_vectors.tolist(),
                        metadatas=batch_metadata,
                        ids=batch_ids,
                    )
                except Exception as e:
                    logger.error(f"Error adding batch {i} to ChromaDB: {e}")
                    raise

                progress.update(task, advance=1)

        logger.info(f"Successfully added {n_samples} vectors from memory-mapped file.")

    def query_similar(
        self,
        query_vector: NDArray[np.float64],
        n_results: int = 20,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Query the database for vectors similar to the query vector.

        Args:
            query_vector: Query latent vector
            n_results: Number of similar vectors to return
            include_metadata: Whether to include orientation metadata in results

        Returns:
            Dictionary with query results including distances and orientations
        """
        if query_vector.ndim > 1:
            query_vector = query_vector.squeeze()

        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Expected query vector of dimension {self.dimension}, got {query_vector.shape[0]}"
            )
        results = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=n_results,
            include=["metadatas", "distances"] if include_metadata else None,
        )
        return results

    def find_best_orientation(
        self,
        query_vector: NDArray[np.float64],
        top_n: int = 20,
        orientation_threshold: float = 1.0,
        min_required_matches: int = 18,
        max_iterations: int = 3,
    ) -> OrientationResult:
        """Find the best matching orientation for a query vector.

        Args:
            query_vector: Query latent vector
            top_n: Number of top matches to consider
            orientation_threshold: Maximum misorientation angle to consider similar
            min_required_matches: Minimum number of similar orientations required
            max_iterations: Maximum number of reference orientations to try

        Returns:
            OrientationResult object containing best fit orientation and related data
        """
        min_required_matches = min(min_required_matches, top_n)
        results = self.query_similar(query_vector, n_results=top_n)
        orientations = np.array(
            [
                [metadata["phi1"], metadata["Phi"], metadata["phi2"]]
                for metadata in results["metadatas"][0]
            ]
        )
        distances = (
            np.array(results["distances"][0]) if "distances" in results else None
        )
        rotations = R.from_euler("zxz", orientations, degrees=True)

        success = False
        best_orientation = orientations[0]
        similar_indices = None
        mean_orientation = None

        # Ensure max_iterations doesn't exceed the number of candidates found
        actual_iterations = min(max_iterations, len(rotations))

        for iteration in range(actual_iterations):
            ref_rotation = rotations[iteration]
            # Vectorized computation of misorientations for all candidates at once
            delta_rotations = ref_rotation.inv() * rotations
            misorientation_angles_rad = delta_rotations.magnitude()
            misorientation_angles_deg = np.degrees(misorientation_angles_rad)

            # Get indices (within the candidate list) of orientations within threshold
            similar_indices = np.where(
                misorientation_angles_deg < orientation_threshold
            )[0]

            if len(similar_indices) >= min_required_matches:
                similar_orientations_euler = []
                # Find symmetry equivalent orientations for all similar candidates
                for idx in similar_indices:
                    # Find the closest symmetry equivalent to the *reference* rotation
                    sym_equivalent_euler = self._find_symmetry_equivalent_orientation(
                        ref_rotation,
                        rotations[idx],  # Use rotations[idx] here too
                    )
                    similar_orientations_euler.append(sym_equivalent_euler)

                # Calculate the mean of the symmetry-adjusted similar orientations
                if similar_orientations_euler:
                    mean_rotation = R.from_euler(
                        "zxz", np.array(similar_orientations_euler), degrees=True
                    ).mean()
                    mean_orientation = mean_rotation.as_euler("zxz", degrees=True)
                else:
                    # This case should ideally not be reached if len(similar_indices) >= min_required_matches
                    mean_orientation = None

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
            # Keep best_orientation as the closest match if no consensus found
            mean_orientation = None  # Ensure mean is None if failed

        return OrientationResult(
            query_vector=query_vector,
            best_orientation=best_orientation,
            mean_orientation=mean_orientation,
            candidate_orientations=orientations,
            distances=distances,
            success=success,
            similar_indices=similar_indices,
        )

    def _find_symmetry_equivalent_orientation(
        self,
        ref_rotation: R,
        candidate_rotation: R,
        quaternion_symmetry_ops: R = QUAT_SYM,
    ) -> NDArray[np.float64]:
        """Find the symmetry equivalent orientation closest to the reference.

        For cubic crystals, there are 24 symmetrically equivalent orientations.
        This function finds the one closest to the reference orientation.

        Args:
            ref_rotation: Reference rotation as a scipy Rotation object
            candidate_rotation: Candidate rotation as a scipy Rotation object
            quaternion_symmetry_ops: Optional tensor of quaternion symmetry operators.
                If None, will use the class's QUAT_SYM attribute

        Returns:
            Symmetrically equivalent orientation in ZXZ Euler angles (degrees)
        """
        # Generate all symmetrically equivalent orientations
        all_sym_rotations = candidate_rotation.inv() * quaternion_symmetry_ops

        # Find which symmetry operation brings candidate closest to reference
        closest_sym_idx = (ref_rotation * all_sym_rotations).magnitude().argmin()

        # Get the orientation in the symmetry frame and return as Euler angles
        sym_equivalent = (
            (all_sym_rotations[closest_sym_idx]).inv().as_euler("zxz", degrees=True)
        )

        return sym_equivalent

    def find_best_orientations_batch(
        self, query_vectors: NDArray[np.float64], batch_size: int = 32, **kwargs
    ) -> list[OrientationResult]:
        """Process multiple query vectors in batches.

        Args:
            query_vectors: Array of query latent vectors
            batch_size: Batch size for processing
            **kwargs: Additional arguments for find_best_orientation

        Returns:
            List of OrientationResult objects for each query vector
        """
        n_vectors = len(query_vectors)
        results = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Finding orientations...", total=n_vectors)

            for i in range(0, n_vectors, batch_size):
                end = min(i + batch_size, n_vectors)
                batch = query_vectors[i:end]

                # Process each vector in the batch
                for j, vector in enumerate(batch):
                    results.append(self.find_best_orientation(vector, **kwargs))
                    progress.update(task, advance=1)

        return results

    def get_count(self) -> int:
        """Get the number of vectors in the database.

        Returns:
            Count of vectors in the database
        """
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the collection from the database."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection '{self.collection_name}'")
