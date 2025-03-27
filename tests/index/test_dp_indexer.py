"""Unit tests for the DiffractionPatternIndexer class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.index.chroma_db import LatentVectorDatabase, OrientationResult
from src.index.dp_indexer import DiffractionPatternIndexer, IndexerConfig


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock VAE model for testing."""
    # Create mock without spec restriction to allow any attributes
    mock_model = MagicMock()
    mock_model.eval.return_value = None

    # Mock encoder and latent space outputs
    mock_encoder_output = MagicMock()
    mock_model.encoder.return_value = mock_encoder_output

    # Mock mu and logvar layers
    mock_model.mu.return_value = torch.randn(64, 16)  # batch_size x latent_dim
    mock_model.logvar.return_value = torch.randn(64, 16)

    # Mock reparameterize to return consistent latent vectors
    mock_model.reparameterize.return_value = torch.randn(64, 16)

    # Mock forward pass to return z, mu, logvar, and decoded
    mock_model.return_value = (
        torch.randn(64, 16),  # z
        torch.randn(64, 16),  # mu
        torch.randn(64, 16),  # logvar
        torch.randn(64, 1, 128, 128),  # decoded
    )

    return mock_model


@pytest.fixture
def mock_db() -> MagicMock:
    """Create a mock LatentVectorDatabase for testing."""
    mock_db = MagicMock(spec=LatentVectorDatabase)

    # Set up find_best_orientation to return predictable results
    mock_result = OrientationResult(
        query_vector=np.random.rand(16),
        best_orientation=np.array([30.0, 45.0, 60.0]),
        mean_orientation=np.array([32.0, 46.0, 61.0]),
        candidate_orientations=np.random.rand(5, 3) * 360,
        distances=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        success=True,
        similar_indices=np.array([0, 1, 2]),
    )
    mock_db.find_best_orientation.return_value = mock_result
    mock_db.find_best_orientations_batch.return_value = [mock_result] * 3

    return mock_db


@pytest.fixture
def test_patterns() -> tuple[torch.Tensor, NDArray[np.float64]]:
    """Create test diffraction patterns as both torch tensor and numpy array."""
    # Create sample patterns: [batch, height, width]
    torch_patterns = torch.rand(3, 128, 128)
    numpy_patterns = torch_patterns.numpy()

    return torch_patterns, numpy_patterns


@pytest.fixture
def test_config() -> IndexerConfig:
    """Create a test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern_path = Path(tmpdir) / "patterns.npy"
        angles_path = Path(tmpdir) / "angles.npy"

        # Create dummy files
        np.save(pattern_path, np.random.rand(10, 128, 128))
        np.save(angles_path, np.random.rand(10, 3) * 360)

        config = IndexerConfig(
            pattern_path=pattern_path,
            angles_path=angles_path,
            batch_size=2,
            device="cpu",
            latent_dim=16,
            image_size=(128, 128),
            top_n=5,
            orientation_threshold=2.0,
        )

        yield config


class TestDiffractionPatternIndexer:
    """Test cases for the DiffractionPatternIndexer class."""

    def test_init(self, mock_model, mock_db, test_config):
        """Test initializing the indexer."""
        # We need to mock build_dictionary to avoid file operations
        with patch.object(DiffractionPatternIndexer, "build_dictionary") as mock_build:
            indexer = DiffractionPatternIndexer(mock_model, mock_db, test_config)

            # Check properties were set correctly
            assert indexer.model is mock_model
            assert indexer.db is mock_db
            assert indexer.config is test_config
            assert str(indexer.device) == "cpu"

            # Check model was prepared
            mock_model.eval.assert_called_once()
            mock_model.to.assert_called_once_with(torch.device("cpu"))
            mock_build.assert_called_once()

    def test_init_fallback_to_cpu(self, mock_model, mock_db, test_config):
        """Test fallback to CPU when CUDA is requested but not available."""
        # Set config to request CUDA
        test_config.device = "cuda"

        # Mock torch.cuda.is_available to return False
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(DiffractionPatternIndexer, "build_dictionary"),
        ):
            indexer = DiffractionPatternIndexer(mock_model, mock_db, test_config)
            assert str(indexer.device) == "cpu"

    def test_build_dictionary(self, mock_model, mock_db, test_config):
        """Test building dictionary from patterns."""
        # Mock the dataloader and latent vector extraction
        mock_dataloader = MagicMock()
        mock_latent = np.random.rand(10, 16)
        mock_orientations = np.random.rand(10, 3) * 360

        with (
            patch.object(
                DiffractionPatternIndexer,
                "_create_dataloader",
                return_value=mock_dataloader,
            ) as mock_create,
            patch.object(
                DiffractionPatternIndexer,
                "_extract_latent_vectors_with_angles",
                return_value=(mock_latent, mock_orientations),
            ) as mock_extract,
        ):
            # Create indexer with init mocked to avoid actual build_dictionary call
            with patch.object(DiffractionPatternIndexer, "__init__", return_value=None):
                indexer = DiffractionPatternIndexer()
                indexer.model = mock_model
                indexer.db = mock_db
                indexer.config = test_config
                indexer.device = torch.device("cpu")

                # Now call build_dictionary explicitly
                indexer.build_dictionary(
                    test_config.pattern_path, test_config.angles_path
                )

                # Check correct methods were called
                mock_create.assert_called_once()
                mock_extract.assert_called_once_with(mock_dataloader)
                mock_db.add_vectors.assert_called_once_with(
                    mock_latent, mock_orientations
                )

    def test_encode_pattern(self, mock_model, mock_db, test_config, test_patterns):
        """Test encoding a single pattern."""
        torch_patterns, numpy_patterns = test_patterns
        single_pattern = torch_patterns[0]  # Single pattern

        with patch.object(DiffractionPatternIndexer, "build_dictionary"):
            indexer = DiffractionPatternIndexer(mock_model, mock_db, test_config)

            # Test with torch tensor
            result = indexer.encode_pattern(single_pattern)
            assert isinstance(result, np.ndarray)

            # Test with numpy array
            with patch(
                "src.index.dp_indexer.create_default_transform",
                return_value=lambda x: torch.tensor(x),
            ):
                result_np = indexer.encode_pattern(numpy_patterns[0])
                assert isinstance(result_np, np.ndarray)

    def test_encode_patterns_batch(
        self, mock_model, mock_db, test_config, test_patterns
    ):
        """Test encoding a batch of patterns."""
        torch_patterns, numpy_patterns = test_patterns

        with patch.object(DiffractionPatternIndexer, "build_dictionary"):
            indexer = DiffractionPatternIndexer(mock_model, mock_db, test_config)

            # Test with torch tensor
            result = indexer.encode_patterns_batch(torch_patterns)
            assert isinstance(result, np.ndarray)

            # Test with numpy array
            with patch(
                "src.index.dp_indexer.create_default_transform",
                return_value=lambda x: torch.tensor(x),
            ):
                result_np = indexer.encode_patterns_batch(numpy_patterns)
                assert isinstance(result_np, np.ndarray)

    def test_index_pattern(self, mock_model, mock_db, test_config, test_patterns):
        """Test indexing a single pattern."""
        torch_patterns, numpy_patterns = test_patterns
        single_pattern = torch_patterns[0]  # Single pattern

        with (
            patch.object(DiffractionPatternIndexer, "build_dictionary"),
            patch.object(
                DiffractionPatternIndexer,
                "encode_pattern",
                return_value=np.random.rand(16),
            ),
        ):
            indexer = DiffractionPatternIndexer(mock_model, mock_db, test_config)

            # Test with default parameters
            result = indexer.index_pattern(single_pattern)
            mock_db.find_best_orientation.assert_called_with(
                ANY,
                top_n=test_config.top_n,
                orientation_threshold=test_config.orientation_threshold,
            )

            # Test with custom parameters
            result = indexer.index_pattern(
                single_pattern, top_n=10, orientation_threshold=1.5
            )
            mock_db.find_best_orientation.assert_called_with(
                ANY, top_n=10, orientation_threshold=1.5
            )

    def test_index_patterns_batch(
        self, mock_model, mock_db, test_config, test_patterns
    ):
        """Test indexing a batch of patterns."""
        torch_patterns, _ = test_patterns

        with (
            patch.object(DiffractionPatternIndexer, "build_dictionary"),
            patch.object(
                DiffractionPatternIndexer,
                "encode_patterns_batch",
                return_value=np.random.rand(3, 16),
            ),
        ):
            indexer = DiffractionPatternIndexer(mock_model, mock_db, test_config)

            # Test batch indexing
            result = indexer.index_patterns_batch(torch_patterns)
            mock_db.find_best_orientations_batch.assert_called_once()

            # Test with custom kwargs
            result = indexer.index_patterns_batch(
                torch_patterns, top_n=10, orientation_threshold=1.5
            )
            mock_db.find_best_orientations_batch.assert_called_with(
                ANY,
                batch_size=test_config.batch_size,
                top_n=10,
                orientation_threshold=1.5,
            )

    def test_extract_latent_vectors_with_angles(self, mock_model, mock_db, test_config):
        """Test extracting latent vectors and angles from dataloader."""
        # Create mock dataloader with 2 batches
        batch1 = (torch.rand(2, 1, 128, 128), torch.rand(2, 3))
        batch2 = (torch.rand(1, 1, 128, 128), torch.rand(1, 3))
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = iter([batch1, batch2])
        mock_dataloader.__len__.return_value = 2

        with patch.object(DiffractionPatternIndexer, "build_dictionary"):
            indexer = DiffractionPatternIndexer(mock_model, mock_db, test_config)

            # Test extraction
            latent_vectors, orientations = indexer._extract_latent_vectors_with_angles(
                mock_dataloader
            )

            # Assertions
            assert isinstance(latent_vectors, np.ndarray)
            assert isinstance(orientations, np.ndarray)
            assert mock_model.encoder.call_count == 2  # Called for each batch
            assert mock_model.mu.call_count == 2
            assert mock_model.logvar.call_count == 2
            assert mock_model.reparameterize.call_count == 2
