"""Unit tests for the LatentVectorDatabase class."""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from src.index.chroma_db import (
    LatentVectorDatabase,
    LatentVectorDatabaseConfig,
    OrientationResult,
)


@pytest.fixture
def mock_chromadb_client() -> tuple[MagicMock, MagicMock]:
    """Create a mock ChromaDB client for testing."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_client.create_collection.return_value = mock_collection
    return mock_client, mock_collection


@pytest.fixture
def test_vectors() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create test vectors and orientations for testing."""
    # Create 5 test vectors with dimension 16
    latent_vectors = np.random.rand(5, 16).astype(np.float64)

    # Create random orientations in ZXZ Euler angles (degrees)
    orientations = np.random.rand(5, 3).astype(np.float64) * 360

    return latent_vectors, orientations


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary directory for database files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def orientation_result() -> OrientationResult:
    """Create a test OrientationResult instance."""
    query_vector = np.random.rand(16).astype(np.float64)
    best_orientation = np.array([30.0, 45.0, 60.0], dtype=np.float64)
    mean_orientation = np.array([32.0, 46.0, 61.0], dtype=np.float64)
    candidate_orientations = np.random.rand(5, 3).astype(np.float64) * 360
    distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    similar_indices = np.array([0, 1, 2], dtype=np.int64)

    return OrientationResult(
        query_vector=query_vector,
        best_orientation=best_orientation,
        mean_orientation=mean_orientation,
        candidate_orientations=candidate_orientations,
        distances=distances,
        success=True,
        similar_indices=similar_indices,
    )


class TestOrientationResult:
    """Test cases for the OrientationResult class."""

    def test_get_top_n_orientations(
        self, orientation_result: OrientationResult
    ) -> None:
        """Test the get_top_n_orientations method."""
        # Test getting top 3 orientations
        top_3 = orientation_result.get_top_n_orientations(3)
        assert top_3.shape == (3, 3)  # 3 orientations, each with 3 angles

        # Check that they're sorted by distance
        sorted_indices = np.argsort(orientation_result.distances)[:3]
        expected_orientations = orientation_result.candidate_orientations[
            sorted_indices
        ]
        np.testing.assert_array_equal(top_3, expected_orientations)

        # Test getting more orientations than available
        top_10 = orientation_result.get_top_n_orientations(10)
        assert top_10.shape == (5, 3)  # Only 5 orientations available

    def test_get_top_n_orientations_no_distances(
        self, orientation_result: OrientationResult
    ) -> None:
        """Test get_top_n_orientations when distances is None."""
        # Create result with None distances
        result_no_dist = OrientationResult(
            query_vector=orientation_result.query_vector,
            best_orientation=orientation_result.best_orientation,
            mean_orientation=orientation_result.mean_orientation,
            candidate_orientations=orientation_result.candidate_orientations,
            distances=None,
            success=True,
            similar_indices=orientation_result.similar_indices,
        )

        top_2 = result_no_dist.get_top_n_orientations(2)
        assert top_2.shape == (2, 3)
        np.testing.assert_array_equal(
            top_2, orientation_result.candidate_orientations[:2]
        )


class TestLatentVectorDatabase:
    """Test cases for the LatentVectorDatabase class."""

    def test_init_in_memory(
        self, mock_chromadb_client: tuple[MagicMock, MagicMock]
    ) -> None:
        """Test initialisation with in-memory database."""
        mock_client, mock_collection = mock_chromadb_client

        with patch("chromadb.Client", return_value=mock_client):
            config = LatentVectorDatabaseConfig(persist_directory=None)
            db = LatentVectorDatabase(config)
            assert db.client == mock_client
            assert db.collection_name == "latent_vectors"
            assert db.dimension == 16
            mock_client.get_collection.assert_called_once_with("latent_vectors")

    def test_init_persistent(
        self, mock_chromadb_client: tuple[MagicMock, MagicMock], temp_db_path: str
    ) -> None:
        """Test initialisation with persistent database."""
        mock_client, mock_collection = mock_chromadb_client

        with patch("chromadb.PersistentClient", return_value=mock_client):
            config = LatentVectorDatabaseConfig(persist_directory=temp_db_path)
            _ = LatentVectorDatabase(config)
            assert Path(temp_db_path).exists()

    def test_init_existing_collection(
        self, mock_chromadb_client: tuple[MagicMock, MagicMock], temp_db_path: str
    ) -> None:
        """Test initialisation with existing collection."""
        mock_client, mock_collection = mock_chromadb_client

        with patch("chromadb.PersistentClient", return_value=mock_client):
            config = LatentVectorDatabaseConfig(persist_directory=temp_db_path)
            _ = LatentVectorDatabase(config)

            mock_client.get_collection.assert_called_once()
            mock_client.create_collection.assert_not_called()

    def test_init_new_collection(
        self, mock_chromadb_client: tuple[MagicMock, MagicMock], temp_db_path: str
    ) -> None:
        """Test initialization with new collection."""
        mock_client, mock_collection = mock_chromadb_client
        mock_client.get_collection.side_effect = ValueError("Collection not found")

        with patch("chromadb.PersistentClient", return_value=mock_client):
            config = LatentVectorDatabaseConfig(persist_directory=temp_db_path)
            _ = LatentVectorDatabase(config)

            mock_client.get_collection.assert_called_once()
            mock_client.create_collection.assert_called_once_with(
                name="latent_vectors", metadata={"dimension": 16}
            )

    def test_validate_vectors_valid(
        self, test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Test vector validation with valid inputs."""
        latent_vectors, orientations = test_vectors

        with patch("chromadb.PersistentClient"):
            db = LatentVectorDatabase()
            # Should not raise any exceptions
            db._validate_vectors(latent_vectors, orientations)

    def test_validate_vectors_mismatched_count(
        self, test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Test vector validation with mismatched counts."""
        latent_vectors, orientations = test_vectors

        with patch("chromadb.PersistentClient"):
            db = LatentVectorDatabase()

            with pytest.raises(
                ValueError, match="Number of latent vectors and orientations must match"
            ):
                db._validate_vectors(latent_vectors, orientations[:-1])

    def test_validate_vectors_wrong_dimension(
        self, test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Test vector validation with incorrect dimensions."""
        latent_vectors, orientations = test_vectors
        wrong_dim_vectors = np.random.rand(5, 8).astype(np.float64)  # 8 instead of 16

        with patch("chromadb.PersistentClient"):
            db = LatentVectorDatabase()

            with pytest.raises(
                ValueError, match="Expected latent vectors of dimension"
            ):
                db._validate_vectors(wrong_dim_vectors, orientations)

    def test_add_vectors(
        self,
        mock_chromadb_client: tuple[MagicMock, MagicMock],
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """Test adding vectors to the database."""
        mock_client, mock_collection = mock_chromadb_client
        latent_vectors, orientations = test_vectors

        # Mock the get_count method to return 0
        mock_collection.count.return_value = 0

        with patch("chromadb.PersistentClient", return_value=mock_client):
            db = LatentVectorDatabase()
            db.add_vectors(latent_vectors, orientations, batch_size=2)

            # Should call add 3 times: 2 batches of 2 and 1 batch of 1
            assert mock_collection.add.call_count == 3

            # Check first batch
            args, kwargs = mock_collection.add.call_args_list[0]
            assert len(kwargs["embeddings"]) == 2
            assert len(kwargs["metadatas"]) == 2
            assert len(kwargs["ids"]) == 2

            # Check IDs format
            assert kwargs["ids"][0] == "vec_0"
            assert kwargs["ids"][1] == "vec_1"

    def test_create_from_files(
        self,
        mock_chromadb_client: tuple[MagicMock, MagicMock],
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        """Test creating database from files."""
        mock_client, mock_collection = mock_chromadb_client
        latent_vectors, orientations = test_vectors

        # Save test vectors to temporary files
        latent_path = Path(temp_db_path) / "latent.npy"
        angles_path = Path(temp_db_path) / "angles.npy"
        np.save(latent_path, latent_vectors)
        np.save(angles_path, orientations)

        with patch("chromadb.PersistentClient", return_value=mock_client):
            db = LatentVectorDatabase()
            db.create_from_files(latent_path, angles_path, batch_size=10)

            # Should call add once with all 5 vectors
            mock_collection.add.assert_called_once()
            args, kwargs = mock_collection.add.call_args
            assert len(kwargs["embeddings"]) == 5
            assert len(kwargs["metadatas"]) == 5
            assert len(kwargs["ids"]) == 5

    def test_query_similar(
        self,
        mock_chromadb_client: tuple[MagicMock, MagicMock],
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """Test querying similar vectors."""
        mock_client, mock_collection = mock_chromadb_client
        latent_vectors, _ = test_vectors

        # Setup mock return value for query
        mock_results = {
            "ids": [["vec_0", "vec_1"]],
            "distances": [[0.1, 0.2]],
            "embeddings": [[latent_vectors[0].tolist(), latent_vectors[1].tolist()]],
            "metadatas": [[{"orientation": [1, 2, 3]}, {"orientation": [4, 5, 6]}]],
        }
        mock_collection.query.return_value = mock_results

        with patch("chromadb.PersistentClient", return_value=mock_client):
            db = LatentVectorDatabase()
            query_vector = latent_vectors[0]
            results = db.query_similar(query_vector, n_results=2)

            mock_collection.query.assert_called_once()
            assert results == mock_results

    def test_query_similar_wrong_dimension(
        self, mock_chromadb_client: tuple[MagicMock, MagicMock]
    ) -> None:
        """Test querying with incorrect vector dimension."""
        mock_client, _ = mock_chromadb_client

        with patch("chromadb.PersistentClient", return_value=mock_client):
            db = LatentVectorDatabase()
            wrong_dim_vector = np.random.rand(8).astype(np.float64)  # 8 instead of 16

            with pytest.raises(ValueError, match="Expected query vector of dimension"):
                db.query_similar(wrong_dim_vector)

    def test_find_best_orientation_simple_example(
        self, mock_chromadb_client: tuple[MagicMock, MagicMock]
    ) -> None:
        """Test the orientation finding with a simple, controlled example."""
        mock_client, mock_collection = mock_chromadb_client

        # Create a simple 16-dimensional query vector
        query_vector = np.ones(16, dtype=np.float64)

        # Create a set of similar orientations (in ZXZ Euler angles, degrees)
        # These are deliberately chosen to be close to each other
        similar_orientations = np.array(
            [
                [30.0, 45.0, 60.0],  # Reference orientation
                [32.0, 44.0, 61.0],  # Small deviation
                [31.0, 46.0, 59.0],  # Small deviation
                [29.0, 45.0, 58.0],  # Small deviation
                [28.0, 43.0, 62.0],  # Small deviation
                [90.0, 90.0, 90.0],  # Outlier - very different
            ],
            dtype=np.float64,
        )

        # Mock query_similar to return our controlled orientations
        query_results = {
            "metadatas": [
                [
                    {"phi1": float(o[0]), "Phi": float(o[1]), "phi2": float(o[2])}
                    for o in similar_orientations
                ]
            ],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.9]],  # Fake distances
        }

        with (
            patch("chromadb.PersistentClient", return_value=mock_client),
            patch.object(
                LatentVectorDatabase, "query_similar", return_value=query_results
            ),
        ):
            db = LatentVectorDatabase()

            # Call with a small threshold that should still find the similar orientations
            # but exclude the outlier
            result = db.find_best_orientation(
                query_vector,
                orientation_threshold=0.3,  # Radians, small enough to exclude outlier
                min_required_matches=3,  # Should find at least the 4 similar ones
                max_iterations=2,
            )

            # Validate the result
            assert isinstance(result, OrientationResult)
            assert result.success is True
            assert result.candidate_orientations.shape == (6, 3)  # All 6 orientations

            # The mean orientation should be close to [30, 45, 60]
            # (not exactly equal due to averaging and possible symmetry operations)
            mean_orientation = result.mean_orientation
            assert 25 < mean_orientation[0] < 35  # phi1 should be around 30
            assert 40 < mean_orientation[1] < 50  # Phi should be around 45
            assert 55 < mean_orientation[2] < 65  # phi2 should be around 60

            # Now test a failure case by requiring more matches than available
            result_failure = db.find_best_orientation(
                query_vector,
                orientation_threshold=0.01,  # Very small threshold
                min_required_matches=5,  # More than we can match
                max_iterations=2,
            )

            # Validate failure result
            assert isinstance(result_failure, OrientationResult)
            assert result_failure.success is False
            assert result_failure.candidate_orientations.shape == (6, 3)
            # In failure case, mean_orientation should be equal to best_orientation
            assert result_failure.mean_orientation is None

    def test_find_best_orientations_batch(
        self,
        mock_chromadb_client: tuple[MagicMock, MagicMock],
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """Test batch processing of orientation finding."""
        mock_client, mock_collection = mock_chromadb_client
        latent_vectors, orientations = test_vectors

        # Create a mock result for find_best_orientation
        mock_result = MagicMock(spec=OrientationResult)

        with (
            patch("chromadb.PersistentClient", return_value=mock_client),
            patch.object(
                LatentVectorDatabase, "find_best_orientation", return_value=mock_result
            ),
        ):
            db = LatentVectorDatabase()

            # Test with 3 vectors and batch_size=2
            results = db.find_best_orientations_batch(latent_vectors[:3], batch_size=2)

            # Should return a list of OrientationResult objects
            assert len(results) == 3
            assert results[0] is mock_result
            assert results[1] is mock_result
            assert results[2] is mock_result

            # Check that find_best_orientation was called 3 times
            assert db.find_best_orientation.call_count == 3
