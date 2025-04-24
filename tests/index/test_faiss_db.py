"""Unit tests for the FaissLatentVectorDatabase class."""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from numpy.typing import NDArray

from latice.index.faiss_db import (
    FaissLatentVectorDatabase,
    FaissLatentVectorDatabaseConfig,
    OrientationResult,
)


@pytest.fixture
def test_vectors() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create test vectors and orientations for testing."""
    latent_vectors = np.random.rand(5, 16).astype(np.float64)
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
        top_3 = orientation_result.get_top_n_orientations(3)
        assert top_3.shape == (3, 3)
        sorted_indices = np.argsort(orientation_result.distances)[:3]
        expected_orientations = orientation_result.candidate_orientations[
            sorted_indices
        ]
        np.testing.assert_array_equal(top_3, expected_orientations)
        top_10 = orientation_result.get_top_n_orientations(10)
        assert top_10.shape == (5, 3)

    def test_get_top_n_orientations_no_distances(
        self, orientation_result: OrientationResult
    ) -> None:
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


class TestFaissLatentVectorDatabase:
    """Test cases for the FaissLatentVectorDatabase class."""

    def test_init(self, temp_db_path: str) -> None:
        config = FaissLatentVectorDatabaseConfig(
            npz_path=str(Path(temp_db_path) / "faiss_index.npz")
        )
        db = FaissLatentVectorDatabase(config)
        assert db.dimension == 16
        assert db.npz_path.exists() is False

    def test_add_vectors(
        self,
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        latent_vectors, orientations = test_vectors
        config = FaissLatentVectorDatabaseConfig(
            npz_path=str(Path(temp_db_path) / "faiss_index.npz")
        )
        db = FaissLatentVectorDatabase(config)
        db.add_vectors(latent_vectors, orientations)
        assert db.get_count() == 5

    def test_create_from_files(
        self,
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        latent_vectors, orientations = test_vectors
        latent_path = Path(temp_db_path) / "latent.npy"
        angles_path = Path(temp_db_path) / "angles.npy"
        np.save(latent_path, latent_vectors)
        np.save(angles_path, orientations)
        config = FaissLatentVectorDatabaseConfig(
            npz_path=str(Path(temp_db_path) / "faiss_index.npz")
        )
        db = FaissLatentVectorDatabase(config)
        db.create_from_files(latent_path, angles_path)
        assert db.get_count() == 5

    def test_query_similar(
        self,
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        latent_vectors, orientations = test_vectors
        config = FaissLatentVectorDatabaseConfig(
            npz_path=str(Path(temp_db_path) / "faiss_index.npz")
        )
        db = FaissLatentVectorDatabase(config)
        db.add_vectors(latent_vectors, orientations)
        query_vector = latent_vectors[0]
        distances, indices = db.query_similar(query_vector, n_results=2)
        assert len(distances) == 2
        assert len(indices) == 2

    def test_query_similar_wrong_dimension(
        self,
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        latent_vectors, orientations = test_vectors
        config = FaissLatentVectorDatabaseConfig(
            npz_path=str(Path(temp_db_path) / "faiss_index.npz")
        )
        db = FaissLatentVectorDatabase(config)
        db.add_vectors(latent_vectors, orientations)
        wrong_dim_vector = np.random.rand(8).astype(np.float64)
        with pytest.raises(ValueError, match="Expected query vector of dimension"):
            db.query_similar(wrong_dim_vector)

    def test_find_best_orientation(
        self,
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        latent_vectors, orientations = test_vectors
        config = FaissLatentVectorDatabaseConfig(
            npz_path=str(Path(temp_db_path) / "faiss_index.npz")
        )
        db = FaissLatentVectorDatabase(config)
        db.add_vectors(latent_vectors, orientations)
        query_vector = latent_vectors[0]
        result = db.find_best_orientation(
            query_vector,
            top_n=3,
            orientation_threshold=10.0,
            min_required_matches=1,
            max_iterations=2,
        )
        assert isinstance(result, OrientationResult)
        assert result.candidate_orientations.shape[1] == 3

    def test_find_best_orientations_batch(
        self,
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        latent_vectors, orientations = test_vectors
        config = FaissLatentVectorDatabaseConfig(
            npz_path=str(Path(temp_db_path) / "faiss_index.npz")
        )
        db = FaissLatentVectorDatabase(config)
        db.add_vectors(latent_vectors, orientations)
        results = db.find_best_orientations_batch(
            latent_vectors[:3], batch_size=2, top_n=2
        )
        assert len(results) == 3
        for res in results:
            assert isinstance(res, OrientationResult)

    def test_save_and_load(
        self,
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        latent_vectors, orientations = test_vectors
        npz_path = Path(temp_db_path) / "faiss_index.npz"
        config = FaissLatentVectorDatabaseConfig(npz_path=str(npz_path))
        db = FaissLatentVectorDatabase(config)
        db.add_vectors(latent_vectors, orientations)
        db.save()
        assert npz_path.exists()
        db2 = FaissLatentVectorDatabase(config)
        assert db2.get_count() == 5

    def test_delete_persistence(
        self,
        test_vectors: tuple[NDArray[np.float64], NDArray[np.float64]],
        temp_db_path: str,
    ) -> None:
        latent_vectors, orientations = test_vectors
        npz_path = Path(temp_db_path) / "faiss_index.npz"
        config = FaissLatentVectorDatabaseConfig(npz_path=str(npz_path))
        db = FaissLatentVectorDatabase(config)
        db.add_vectors(latent_vectors, orientations)
        db.save()
        assert npz_path.exists()
        db.delete_persistence()
        assert not npz_path.exists()
