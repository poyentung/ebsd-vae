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
    """

    collection_name: str = "latent_vectors"
    persist_directory: str | None = ".chroma_db"
    dimension: int = 16


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

    def _validate_vectors(
        self, latent_vectors: NDArray[np.float64], orientations: NDArray[np.float64]
    ) -> None:
        if len(latent_vectors) != len(orientations):
            raise ValueError("Number of latent vectors and orientations must match")

        if latent_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected latent vectors of dimension {self.dimension}, got {latent_vectors.shape[1]}"
            )

    def add_vectors(
        self,
        latent_vectors: NDArray[np.float64],
        orientations: NDArray[np.float64],
        batch_size: int = 1000,
    ) -> None:
        """Add latent vectors and their orientations to the database.

        Args:
            latent_vectors: Array of latent vectors (shape: n_samples x dimension)
            orientations: Array of orientation vectors (shape: n_samples x 3)
            batch_size: Maximum number of vectors to add in a single batch
        """
        self._validate_vectors(latent_vectors, orientations)

        # Get current count to offset IDs
        current_count = self.get_count()

        # Process in batches to avoid memory issues
        n_samples = len(latent_vectors)
        n_batches = (n_samples + batch_size - 1) // batch_size

        logger.info(
            f"Adding {n_samples} vectors to collection '{self.collection_name}' in {n_batches} batches"
        )

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Adding vectors to database...", total=n_batches)

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                # Prepare batch data with offset IDs
                batch_ids = [
                    f"vec_{j + current_count}" for j in range(start_idx, end_idx)
                ]
                batch_vectors = latent_vectors[start_idx:end_idx].tolist()
                batch_orientations = orientations[start_idx:end_idx].tolist()

                # Convert orientations to strings or individual components
                batch_metadata = [
                    {
                        "orientation_str": ",".join(map(str, orient)),
                        "phi1": float(orient[0]),
                        "Phi": float(orient[1]),
                        "phi2": float(orient[2]),
                    }
                    for orient in batch_orientations
                ]

                # Add to collection
                self.collection.add(
                    embeddings=batch_vectors, metadatas=batch_metadata, ids=batch_ids
                )

                # Update progress
                progress.update(task, advance=1)

        logger.info(f"Successfully added {n_samples} vectors to the database")

    def create_from_files(
        self, latent_file_path: Path, angles_file_path: Path, batch_size: int = 1000
    ) -> None:
        """Create the vector database from latent and angle files.

        Args:
            latent_file_path: Path to the numpy file containing latent vectors
            angles_file_path: Path to the numpy file containing orientation angles
            batch_size: Maximum batch size for adding vectors
        """
        latent_file_path = Path(latent_file_path)
        angles_file_path = Path(angles_file_path)

        logger.info(f"Loading latent vectors from {latent_file_path}")
        latent_vectors = np.load(latent_file_path)

        logger.info(f"Loading orientations from {angles_file_path}")
        orientations = np.load(angles_file_path)

        self.add_vectors(latent_vectors, orientations, batch_size)

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
        results = self.query_similar(query_vector, n_results=top_n)

        # Reconstruct orientations from metadata components
        orientations = np.array(
            [
                [metadata["phi1"], metadata["Phi"], metadata["phi2"]]
                for metadata in results["metadatas"][0]
            ]
        )

        # Extract distances
        distances = (
            np.array(results["distances"][0]) if "distances" in results else None
        )

        rotations = R.from_euler("zxz", orientations, degrees=True)

        success = False
        best_orientation = orientations[0]
        similar_indices = None

        for iteration in range(max_iterations):
            ref_rotation = R.from_euler("zxz", orientations[iteration], degrees=True)
            similar_orientations = []

            # Vectorized computation of misorientations for all candidates at once
            misorientation_angles = (ref_rotation * rotations.inv()).magnitude()

            # Get indices of orientations within threshold
            similar_indices = np.where(misorientation_angles < orientation_threshold)[0]
            if len(similar_indices) >= min_required_matches:
                # Find symmetry equivalent orientations for all similar candidates at once
                for idx in similar_indices:
                    # Find the closest symmetry equivalent
                    sym_equivalent = self._find_symmetry_equivalent_orientation(
                        ref_rotation, rotations[idx]
                    )
                    similar_orientations.append(sym_equivalent)

                mean_orientation = (
                    R.from_euler("zxz", np.array(similar_orientations), degrees=True)
                    .mean()
                    .as_euler("zxz", degrees=True)
                )
                success = True
                break

        if not success:
            mean_orientation = None
            logger.warning(
                f"Failed to find best orientation after {max_iterations} iterations"
            )

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
