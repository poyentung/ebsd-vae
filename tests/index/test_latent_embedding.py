import pytest
from pathlib import Path
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field

from latice.index.latent_embedding import (
    IndexerConfig,
    LatentVectorDataset,
    DiffractionPatternIndexer,
)
from latice.index.chroma_db import LatentVectorDatabaseConfig
from latice.model import VariationalAutoEncoder
from latice.utils.utils import get_device


@pytest.fixture
def mock_model():
    """Create a mock VAE model for testing."""
    mock = Mock(spec=VariationalAutoEncoder)
    mock.encoder = Mock(return_value=torch.randn(2, 32, 8, 8))
    mock.mu = Mock(return_value=torch.randn(2, 16))
    mock.logvar = Mock(return_value=torch.randn(2, 16))
    mock.reparameterize = Mock(return_value=torch.randn(2, 16))
    mock.eval = Mock()
    mock.to = Mock(return_value=mock)
    return mock


@pytest.fixture
def mock_data_module():
    """Create a mock data module for testing."""
    mock = Mock()
    # Create mock dataset with 10 patterns
    mock.dataset_full = Mock()
    mock.dataset_full.ebsp_dataset = np.random.random((10, 128, 128))

    # Mock getitem to return a pattern and empty orientation
    mock.dataset_full.__getitem__ = Mock(
        return_value=(torch.randn(1, 128, 128), torch.zeros(3))
    )
    return mock


@pytest.fixture
def temp_vector_file(tmp_path):
    """Create a temporary latent vector file for testing."""
    vectors = np.random.random((5, 16)).astype(np.float32)
    file_path = tmp_path / "test_vectors.npy"
    np.save(file_path, vectors)
    return file_path


@dataclass
class IndexerConfig:
    """Configuration parameters for the diffraction pattern indexer.

    This dataclass encapsulates all configuration parameters needed for
    pattern indexing, model configuration, and data processing.

    Attributes:
        val_data_ratio: Ratio of validation data to total data
        batch_size: Batch size for model training and inference
        n_cpu: Number of CPU cores to use for data loading
        image_size: Size of input images (height, width)
        inplanes: Number of input planes for the model
        learning_rate: Learning rate for optimizer
        decay: Weight decay for optimizer
        kl_lambda: Weight for KL divergence loss term
        optimizer: Optimizer type ("adam", "sgd", etc.)
        lr_scheduler_kw: Optional learning rate scheduler parameters
        model: VAE model class
        random_seed: Random seed for reproducibility
    """

    batch_size: int = 64
    n_cpu: int = 4
    image_size: tuple[int, int] = (128, 128)
    inplanes: int = 32
    model: VariationalAutoEncoder
    db_config: LatentVectorDatabaseConfig = field(
        default_factory=LatentVectorDatabaseConfig
    )


class TestIndexerConfig:
    """Tests for the IndexerConfig class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        model_mock = Mock(spec=VariationalAutoEncoder)
        config = IndexerConfig(model=model_mock)

        assert config.batch_size == 64
        assert config.n_cpu == 4
        assert config.image_size == (128, 128)
        assert config.inplanes == 32
        assert config.model is model_mock
        assert isinstance(config.db_config, LatentVectorDatabaseConfig)

    def test_custom_values(self):
        """Test that custom values are set correctly."""
        model_mock = Mock(spec=VariationalAutoEncoder)
        db_config = LatentVectorDatabaseConfig(collection_name="test_collection")

        config = IndexerConfig(
            batch_size=32,
            n_cpu=2,
            image_size=(64, 64),
            inplanes=16,
            model=model_mock,
            db_config=db_config,
        )

        assert config.batch_size == 32
        assert config.n_cpu == 2
        assert config.image_size == (64, 64)
        assert config.inplanes == 16
        assert config.model is model_mock
        assert config.db_config is db_config
        assert config.db_config.collection_name == "test_collection"


class TestLatentVectorDataset:
    """Tests for the LatentVectorDataset class."""

    def test_initialization(self, temp_vector_file):
        """Test that dataset is correctly initialized."""
        device = torch.device("cpu")
        dataset = LatentVectorDataset(temp_vector_file, device)

        # Check if vectors were loaded correctly
        assert isinstance(dataset.vectors, torch.Tensor)
        assert dataset.vectors.shape == (5, 16)
        assert dataset.vectors.device == device

    def test_len(self, temp_vector_file):
        """Test the __len__ method."""
        device = torch.device("cpu")
        dataset = LatentVectorDataset(temp_vector_file, device)

        assert len(dataset) == 5

    def test_getitem(self, temp_vector_file):
        """Test the __getitem__ method."""
        device = torch.device("cpu")
        dataset = LatentVectorDataset(temp_vector_file, device)

        # Get first item
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (16,)

        # Get last item
        item = dataset[4]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (16,)


class TestDiffractionPatternIndexer:
    """Tests for the DiffractionPatternIndexer class."""

    def test_initialization(self, mock_model):
        """Test that indexer is correctly initialized."""
        config = IndexerConfig(model=mock_model)

        with patch("latice.index.latent_embedding.LatentVectorDatabase") as mock_db:
            indexer = DiffractionPatternIndexer(config=config)

            # Check if model is set to eval mode and moved to device
            mock_model.eval.assert_called_once()
            mock_model.to.assert_called_once()

            # Check if database was created
            mock_db.assert_called_once()

    def test_device_selection(self, mock_model):
        """Test that device is correctly selected."""
        config = IndexerConfig(model=mock_model)

        # Test with explicit CPU device
        with patch("latice.index.latent_embedding.LatentVectorDatabase"):
            cpu_device = torch.device("cpu")
            indexer = DiffractionPatternIndexer(config=config, device=cpu_device)
            assert indexer.device == cpu_device

        # Test with default device selection
        with patch("latice.index.latent_embedding.LatentVectorDatabase"):
            with patch("torch.cuda.is_available", return_value=True):
                indexer = DiffractionPatternIndexer(config=config)
                assert indexer.device == torch.device("cuda")

            with patch("torch.cuda.is_available", return_value=False):
                indexer = DiffractionPatternIndexer(config=config)
                assert indexer.device == torch.device("cpu")

    def test_generate_latent_vectors(self, mock_model, mock_data_module, tmp_path):
        """Test latent vector generation."""
        config = IndexerConfig(model=mock_model)
        pattern_path = Path(tmp_path / "patterns.npy")
        angles_path = Path(tmp_path / "angles.txt")
        output_path = Path(tmp_path / "latent_vectors.npy")

        with patch("latice.index.latent_embedding.LatentVectorDatabase"):
            indexer = DiffractionPatternIndexer(config=config)

            # Mock _create_data_module to return our mock data module
            indexer._create_data_module = Mock(return_value=mock_data_module)

            # Test without output path
            latent_vectors = indexer.generate_latent_vectors(pattern_path, angles_path)
            assert isinstance(latent_vectors, np.ndarray)

            # Test with output path
            latent_vectors = indexer.generate_latent_vectors(
                pattern_path, angles_path, output_path
            )
            assert isinstance(latent_vectors, np.ndarray)
            assert output_path.exists()

    def test_extract_latent_vectors(self, mock_model, mock_data_module):
        """Test latent vector extraction."""
        config = IndexerConfig(model=mock_model)

        with patch("latice.index.latent_embedding.LatentVectorDatabase"):
            indexer = DiffractionPatternIndexer(config=config)

            latent_vectors = indexer._extract_latent_vectors(
                mock_model, mock_data_module
            )

            # Check if the model was called correctly
            assert mock_model.encoder.call_count == 10
            assert mock_model.mu.call_count == 10
            assert mock_model.logvar.call_count == 10
            assert mock_model.reparameterize.call_count == 10

            # Check the result format
            assert isinstance(latent_vectors, list)
            assert len(latent_vectors) == 10
            for vec in latent_vectors:
                assert isinstance(vec, np.ndarray)

    def test_create_data_module(self, mock_model):
        """Test data module creation."""
        config = IndexerConfig(model=mock_model)
        pattern_path = Path("patterns.npy")
        angles_path = Path("angles.txt")

        with patch("latice.index.latent_embedding.LatentVectorDatabase"):
            with patch("latice.index.latent_embedding.DPDataModule") as mock_dm_class:
                indexer = DiffractionPatternIndexer(config=config)
                indexer._create_data_module(pattern_path, angles_path)

                # Check if DPDataModule was created with correct parameters
                mock_dm_class.assert_called_once_with(
                    path=pattern_path,
                    rot_angles_path=angles_path,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                )


# Add device compatibility tests
def test_device_compatibility():
    """Test the device compatibility across different platforms."""
    model_mock = Mock(spec=VariationalAutoEncoder)
    config = IndexerConfig(model=model_mock)

    with patch("latice.index.latent_embedding.LatentVectorDatabase"):
        # Test CUDA device
        with patch("torch.cuda.is_available", return_value=True):
            indexer = DiffractionPatternIndexer(config=config)
            assert indexer.device == torch.device("cuda")

        # Test MPS device (for Apple Silicon)
        with patch("torch.cuda.is_available", return_value=False):
            # Create a mock for torch.backends.mps
            mps_mock = MagicMock()
            mps_mock.is_available.return_value = True

            with patch.object(torch, "backends", MagicMock(mps=mps_mock)):
                # This would need to be implemented in the actual code
                # Here we're testing a hypothetical implementation that checks for MPS
                with patch(
                    "latice.utils.utils.get_device", return_value=torch.device("mps")
                ):
                    indexer = DiffractionPatternIndexer(config=config)
                    assert indexer.device == torch.device("mps")

        # Test CPU fallback
        with patch("torch.cuda.is_available", return_value=False):
            indexer = DiffractionPatternIndexer(config=config)
            assert indexer.device == torch.device("cpu")
