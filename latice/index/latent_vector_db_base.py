from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class OrientationResult:
    """Results from orientation matching query.

    This class encapsulates the complete results of an orientation matching query,
    including the original query vector, best matched orientation, and all candidate
    orientations with their similarity metrics.

    Attributes:
        query_vector: Original latent vector used for the query.
        best_orientation: Best matched orientation in ZXZ Euler angles (degrees).
        candidate_orientations: All top candidate orientations from similarity search.
        distances: Distance metrics for each candidate orientation.
        success: Whether a valid orientation match was found.
        similar_indices: Indices of orientations within the similarity threshold.
    """

    query_vector: NDArray[np.float64]
    best_orientation: NDArray[np.float64]
    candidate_orientations: NDArray[np.float64]
    distances: NDArray[np.float64]
    mean_orientation: NDArray[np.float64] | None = None
    success: bool = True
    similar_indices: NDArray[np.int64] = None

    def get_top_n_orientations(self, n: int = 5) -> NDArray[np.float64]:
        """Return the top N orientations sorted by similarity.

        Args:
            n: Number of top orientations to return.

        Returns:
            Array of top N orientations in ZXZ Euler angles (degrees).
        """
        if self.distances is None or len(self.distances) == 0:
            return self.candidate_orientations[
                : min(n, len(self.candidate_orientations))
            ]

        # Sort orientations by distance
        sorted_indices = np.argsort(self.distances)
        return self.candidate_orientations[
            sorted_indices[: min(n, len(sorted_indices))]
        ]


class LatentVectorDatabaseBase(ABC):
    """Abstract base class for latent vector databases.

    This class defines the interface for vector databases that store and query latent vectors and their orientations.
    Implementations must provide methods for adding vectors, querying similar vectors, finding best orientations, and batch operations.
    """

    @abstractmethod
    def add_vectors(
        self,
        latent_vectors: NDArray[np.float64],
        orientations: NDArray[np.float64],
        **kwargs: Any,
    ) -> None:
        """Add latent vectors and their orientations to the database."""
        pass

    @abstractmethod
    def create_from_files(
        self, latent_file_path: Path | str, angles_file_path: Path | str, **kwargs: Any
    ) -> None:
        """Create the vector database from latent and angle files."""
        pass

    @abstractmethod
    def query_similar(
        self, query_vector: NDArray[np.float64], n_results: int = 20, **kwargs: Any
    ) -> Any:
        """Query the database for vectors similar to the query vector."""
        pass

    @abstractmethod
    def find_best_orientation(
        self, query_vector: NDArray[np.float64], **kwargs: Any
    ) -> Any:
        """Find the best matching orientation for a query vector."""
        pass

    @abstractmethod
    def find_best_orientations_batch(
        self, query_vectors: NDArray[np.float64], **kwargs: Any
    ) -> list[Any]:
        """Process multiple query vectors in batches."""
        pass

    @abstractmethod
    def get_count(self) -> int:
        """Get the number of vectors in the database."""
        pass
