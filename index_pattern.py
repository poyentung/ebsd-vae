import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from src.data_module import DPDataModule
from src.model import VariationalAutoEncoderRawData
from src.lightning_module import VAELightningModule
from src.utils.constants import CUBIC_SYMMETRY

logger = logging.getLogger(__name__)

VAE_MODEL_PATH = Path("trained_models/checkpoints/vae_final.pth")
SAMPLE_PATTERN_PATH = Path("data/sample_pattern.npy")
SAMPLE_ANGFILE_PATH = Path("data/anglefile_sample.txt")
DICTIONARY_LATENT_PATH = Path("data/dic_latent.npy")
DICTIONARY_ANGLES_PATH = Path("data/dic_ang.npy")


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

    val_data_ratio: float = 0.1
    batch_size: int = 64
    n_cpu: int = 4
    image_size: Tuple[int, int] = (128, 128)
    inplanes: int = 32
    learning_rate: float = 1e-4
    decay: float = 2.5e-4
    kl_lambda: float = 5e-6
    optimizer: str = "adam"
    lr_scheduler_kw: Optional[dict] = None
    model: Type[Any] = VariationalAutoEncoderRawData
    random_seed: int = 42
    # Add any additional paths with defaults
    save_dir: str = "austenite_100_gaussian"
    model_name: str = "VAE_base_line"
    precision_for_training: int = 16


class LatentVectorDataset(Dataset):
    """Dataset class for loading and processing latent vectors.

    Attributes:
        vectors: Tensor containing latent vectors
    """

    def __init__(self, latent_file_path: Path, device: torch.device) -> None:
        """Initialize the dataset with latent vectors from a numpy file.

        Args:
            latent_file_path: Path to the numpy file containing latent vectors
            device: Device to load the tensors to (CPU, MPS or GPU)
        """
        vectors = np.load(latent_file_path)
        self.vectors = torch.from_numpy(vectors).to(device)

    def __len__(self) -> int:
        return len(self.vectors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]


class DiffractionPatternIndexer:
    """Class for indexing electron diffraction patterns using a VAE model.

    This class handles the process of encoding diffraction patterns into
    a latent space and finding the best matching orientations from a
    dictionary of pre-computed patterns.

    Attributes:
        model_path: Path to the trained VAE model
        device: Device to use for computation (CPU or GPU)
        config: Configuration parameters for the model and data processing
    """

    def __init__(
        self,
        model_path: Path = Path("trained_models/checkpoints/vae_final.pth"),
        device: torch.device | None = None,
        config: Optional[IndexerConfig] = None,
    ) -> None:
        """Initialize the diffraction pattern indexer.

        Args:
            model_path: Path to the trained VAE model
            device: Device to use for computation; if None, uses GPU if available
            config: Configuration parameters (default configuration used if None)
        """
        # Initialize configuration
        self.config = config if config is not None else IndexerConfig()

        # Set random seeds for reproducibility
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        self.model_path = Path(model_path)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Using device: {self.device}")

    def generate_latent_vectors(
        self, pattern_path: Path, angles_path: Path, output_path: Path | None = None
    ) -> NDArray[np.float32]:
        """Generate latent vectors from diffraction patterns.

        Args:
            pattern_path: Path to the patterns numpy file
            angles_path: Path to the rotation angles file
            output_path: Optional path to save the generated latent vectors

        Returns:
            Normalized latent vectors as a numpy array
        """
        model = self._load_vae_model()
        data_module = self._create_data_module(pattern_path, angles_path)
        latent_vectors = self._extract_latent_vectors(model, data_module)

        latent_array = np.array(latent_vectors)
        vector_norms = np.linalg.norm(latent_array, axis=1, keepdims=True)
        normalised_vectors = np.divide(
            latent_array,
            vector_norms,
            out=np.zeros_like(latent_array),
            where=vector_norms != 0,
        )

        if output_path:
            np.save(output_path, normalised_vectors)
            logger.info(f"Latent vectors saved to {output_path}")

        return normalised_vectors

    def _load_vae_model(self) -> VAELightningModule:
        """Load and prepare the VAE model for inference."""
        model = VAELightningModule(model=self.config.model).load_from_checkpoint(
            self.model_path
        )
        model.eval()
        model.to(self.device)
        return model

    def _create_data_module(
        self, pattern_path: Path, rot_angles_path: Path
    ) -> DPDataModule:
        """Create a data module for the given pattern and rotation angles."""
        return DPDataModule(
            path=pattern_path,
            rot_angles_path=rot_angles_path,
            image_size=self.config.image_size,
            val_data_ratio=self.config.val_data_ratio,
            batch_size=self.config.batch_size,
            n_cpu=self.config.n_cpu,
            seed=self.config.random_seed,
        )

    def _extract_latent_vectors(
        self, model: VAELightningModule, data_module: DPDataModule
    ) -> list[NDArray]:
        """Extract latent vectors from patterns using the VAE model."""
        latent_vectors = []
        total_patterns = data_module.dataset_full.ebsp_dataset.shape[0]
        logger.info(f"Processing {total_patterns} patterns...")

        with torch.no_grad():
            for i in range(total_patterns):
                data, _ = data_module.dataset_full[i]
                data = data.unsqueeze(0).to(self.device)

                # Encode pattern to latent space
                enc = model.model.encoder(data)
                mu = model.model.mu(enc.flatten(1, -1))
                logvar = model.model.logvar(enc.flatten(1, -1))
                z = model.model.reparameterize(mu, logvar)
                latent_vectors.append(z.cpu().numpy().squeeze())

        return latent_vectors

    def find_best_orientations(
        self,
        dict_latent_path: Path,
        exp_latent_path: Path,
        dict_angles_path: Path,
        output_path: Path | None = None,
        top_n: int = 20,
        batch_size: int = 1024,
        patterns_per_batch: int = 64,
    ) -> np.ndarray:
        """Find the best matching orientations for experimental patterns.

        Args:
            dict_latent_path: Path to the dictionary latent vectors
            exp_latent_path: Path to the experimental latent vectors
            dict_angles_path: Path to the dictionary angles
            output_path: Optional path to save the results
            top_n: Number of top matches to consider
            batch_size: Batch size for dictionary processing
            patterns_per_batch: Number of experimental patterns to process in each batch

        Returns:
            Best fit angles as a numpy array
        """
        # Load datasets and create dataloaders
        dict_latent, exp_latent, dict_angles, dataloaders = (
            self._prepare_data_for_matching(
                dict_latent_path,
                exp_latent_path,
                dict_angles_path,
                batch_size,
                patterns_per_batch,
            )
        )

        # Process each batch of experimental patterns
        best_fit_angles = self._process_experimental_batches(
            dataloaders, dict_angles, top_n, patterns_per_batch
        )

        # Save results if requested
        if output_path:
            np.save(output_path, best_fit_angles)
            logger.info(f"Indexing results saved to {output_path}")

        return best_fit_angles

    def _prepare_data_for_matching(
        self,
        dict_latent_path: Path,
        exp_latent_path: Path,
        dict_angles_path: Path,
        batch_size: int,
        patterns_per_batch: int,
    ) -> tuple:
        """Prepare datasets and dataloaders for orientation matching.

        Args:
            dict_latent_path: Path to dictionary latent vectors
            exp_latent_path: Path to experimental latent vectors
            dict_angles_path: Path to dictionary angles
            batch_size: Batch size for dictionary processing
            patterns_per_batch: Number of experimental patterns per batch

        Returns:
            Tuple containing (dict_latent, exp_latent, dict_angles, dataloaders)
        """
        # Load datasets
        dict_latent = LatentVectorDataset(dict_latent_path, self.device)
        exp_latent = LatentVectorDataset(exp_latent_path, self.device)
        dict_angles = np.load(dict_angles_path)

        # Create dataloaders
        dict_dataloader = DataLoader(dict_latent, batch_size=batch_size, shuffle=False)
        exp_dataloader = DataLoader(
            exp_latent, batch_size=patterns_per_batch, shuffle=False
        )

        dataloaders = {"dictionary": dict_dataloader, "experimental": exp_dataloader}

        return dict_latent, exp_latent, dict_angles, dataloaders

    def _process_experimental_batches(
        self,
        dataloaders: dict,
        dict_angles: np.ndarray,
        top_n: int,
        patterns_per_batch: int,
    ) -> np.ndarray:
        """Process batches of experimental patterns to find orientations.

        Args:
            dataloaders: Dictionary containing dataloaders for dictionary and experimental data
            dict_angles: Array of dictionary angles
            top_n: Number of top matches to consider
            patterns_per_batch: Number of patterns per batch for logging

        Returns:
            Array of best fit angles for all experimental patterns
        """
        dict_dataloader = dataloaders["dictionary"]
        exp_dataloader = dataloaders["experimental"]

        best_fit_angles = []
        batch_count = 0

        for exp_batch in exp_dataloader:
            batch_start = batch_count * patterns_per_batch
            batch_end = min(
                batch_start + exp_batch.shape[0] - 1,
                batch_start + patterns_per_batch - 1,
            )

            logger.info(
                f"Searching best matches for patterns {batch_start} to {batch_end}..."
            )

            # Compute similarities and get top indices
            top_indices = self._compute_similarities(exp_batch, dict_dataloader, top_n)

            # Process each pattern in the batch
            for i in range(exp_batch.shape[0]):
                best_angle = self._process_top_matches(
                    dict_angles, top_indices[i], top_n
                )
                best_fit_angles.append(best_angle)

            batch_count += 1

        return np.array(best_fit_angles)

    def _compute_similarities(
        self, exp_batch: torch.Tensor, dict_dataloader: DataLoader, top_n: int
    ) -> np.ndarray:
        """Compute similarities between experimental and dictionary patterns.

        Args:
            exp_batch: Batch of experimental latent vectors
            dict_dataloader: DataLoader for dictionary latent vectors
            top_n: Number of top matches to return

        Returns:
            Array of indices of top matching patterns for each experimental pattern
        """
        # Initialize similarity matrix
        all_similarities = torch.zeros((exp_batch.shape[0], 0), dtype=torch.float64).to(
            self.device
        )

        # Calculate similarities with all dictionary patterns in batches
        for dict_batch in dict_dataloader:
            dict_batch = dict_batch.to(self.device)
            # Compute dot product between experimental and dictionary latent vectors
            batch_similarities = torch.mm(
                exp_batch.squeeze(), dict_batch.squeeze().transpose(0, 1)
            )
            all_similarities = torch.cat((all_similarities, batch_similarities), 1)

        # Get indices of top matches
        _, indices = torch.sort(all_similarities, descending=True)
        top_indices = indices.cpu().numpy()[:, :top_n]

        return top_indices

    def _process_top_matches(
        self,
        dictionary_angles: np.ndarray,
        top_match_indices: np.ndarray,
        top_n: int,
        max_iterations: int = 3,
        orientation_threshold: float = 1.0,
        min_required_matches: int = 18,
    ) -> np.ndarray:
        """Process top matches to find the best orientation considering symmetry.

        This function iteratively tries reference orientations from the top matches,
        finding all orientations that are similar within symmetry considerations.
        When enough similar orientations are found, their average is computed.

        Args:
            dictionary_angles: Array of dictionary angles in ZXZ Euler representation
            top_match_indices: Indices of top matching patterns sorted by similarity
            num_top_matches: Number of top matches to consider
            max_iterations: Maximum number of reference orientations to try
            orientation_threshold: Maximum misorientation angle (in radians) to consider similar
            min_required_matches: Minimum number of similar orientations required for a valid match

        Returns:
            Best fit orientation in ZXZ Euler angles (degrees)
        """
        for iteration in range(max_iterations):
            # Select a reference orientation from top matches
            reference_orientation = dictionary_angles[top_match_indices[iteration]]
            similar_orientations = []

            # Find all orientations similar to the reference orientation
            for match_idx in range(top_n):
                candidate_orientation = dictionary_angles[top_match_indices[match_idx]]

                # Find symmetrically equivalent orientation closest to reference
                symmetric_orientation = self._find_symmetry_equivalent_orientation(
                    reference_orientation, candidate_orientation
                )

                # Check if orientations are similar enough (within threshold)
                misorientation_angle = self._calculate_misorientation(
                    reference_orientation, candidate_orientation
                )

                if misorientation_angle < orientation_threshold:
                    similar_orientations.append(symmetric_orientation)

            # If enough similar orientations found, calculate average and return
            if (
                len(similar_orientations) >= min_required_matches
                or iteration == max_iterations - 1
            ):
                similar_orientations_array = np.array(similar_orientations)

                # Calculate average orientation in quaternion space
                mean_orientation = (
                    R.from_euler("zxz", similar_orientations_array, degrees=True)
                    .mean()
                    .as_euler("zxz", degrees=True)
                )

                if iteration > 0:
                    ordinal_suffix = self._get_ordinal_suffix(iteration + 1)
                    logger.info(
                        f"Used {iteration + 1}{ordinal_suffix} best match as reference after "
                        f"{iteration} outlier detection attempts"
                    )

                return mean_orientation

            # If not enough matches, try next best orientation
            ordinal_suffix = self._get_ordinal_suffix(iteration + 2)
            logger.info(
                f"Outlier detected. Trying {iteration + 2}{ordinal_suffix} best match as reference"
            )

        # If all iterations fail, return the orientation from the last attempt
        # This should never happen if max_iterations is properly set
        return mean_orientation

    def _find_symmetry_equivalent_orientation(
        self, reference: np.ndarray, candidate: np.ndarray
    ) -> np.ndarray:
        """Find the symmetry equivalent orientation closest to the reference.

        For cubic crystals, there are 24 symmetrically equivalent orientations.
        This function finds the one closest to the reference orientation.

        Args:
            reference: Reference orientation in ZXZ Euler angles (degrees)
            candidate: Candidate orientation in ZXZ Euler angles (degrees)

        Returns:
            Symmetrically equivalent orientation in ZXZ Euler angles (degrees)
        """
        # Convert to rotation objects
        ref_rotation = R.from_euler("zxz", reference, degrees=True)
        cand_rotation = R.from_euler("zxz", candidate, degrees=True)

        # Generate all 24 symmetrically equivalent orientations
        all_sym_rotations = cand_rotation.inv() * R.from_quat(CUBIC_SYMMETRY)

        # Find which symmetry operation brings candidate closest to reference
        closest_sym_idx = (ref_rotation * all_sym_rotations).magnitude().argmin()

        # Get the orientation in the symmetry frame and return as Euler angles
        sym_equivalent = (
            (all_sym_rotations[closest_sym_idx]).inv().as_euler("zxz", degrees=True)
        )

        return sym_equivalent

    def _calculate_misorientation(
        self, orientation1: np.ndarray, orientation2: np.ndarray
    ) -> float:
        """Calculate the misorientation angle between two orientations.

        Args:
            orientation1: First orientation in ZXZ Euler angles (degrees)
            orientation2: Second orientation in ZXZ Euler angles (degrees)

        Returns:
            Misorientation angle in radians
        """
        rot1 = R.from_euler("zxz", orientation1, degrees=True)
        rot2 = R.from_euler("zxz", orientation2, degrees=True)

        # Calculate misorientation
        misorientation = (rot1 * rot2.inv()).magnitude()

        return misorientation

    def _get_ordinal_suffix(self, n: int) -> str:
        """Return the ordinal suffix for a number.

        Args:
            n: The number

        Returns:
            The ordinal suffix (st, nd, rd, or th)
        """
        if 11 <= (n % 100) <= 13:
            return "th"

        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return suffix


# # Run indexing
# best_fit_angle = get_best_fit(
#     dic=DICTIONARY_LATENT_PATH,
#     exp="exp_latent_sample.npy",
#     dic_ang=DICTIONARY_ANGLES_PATH,
#     top_N=20,
#     batch_size=1024,
# )
# np.save("best_fit_results.npy", best_fit_angle)
# print("Indexing complete. Results saved to 'best_fit_results.npy'")
