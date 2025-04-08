from pathlib import Path
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_default_transform(image_size: tuple[int, int]) -> transforms.Compose:
    """Create default transform pipeline with specified image size.

    Args:
        image_size: Tuple of (height, width) for the output image

    Returns:
        Composed transform pipeline
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
        ]
    )


class DPdataset(Dataset):
    """Diffraction Pattern dataset for electron backscatter patterns.

    This dataset loads electron backscatter diffraction patterns (EBSP) and their
    corresponding rotation angles for training orientation estimation models.

    Attributes:
        ebsp_dataset: Numpy array containing the diffraction patterns.
        rot_angles: Pandas DataFrame containing rotation angles (z1, x, z2).
        transform: Torchvision transforms to apply to the images.
    """

    def __init__(
        self,
        path: str | Path,
        rot_angles_path: str | Path,
        image_size: tuple[int, int] = (128, 128),
        transform: transforms.Compose | None = None,
    ) -> None:
        """Initialize the Diffraction Pattern dataset.

        Args:
            path: Path to the .npy file containing diffraction patterns.
            rot_angles_path: Path to the file containing rotation angles.
            image_size: Size to crop the images to (height, width).

        Raises:
            ValueError: If the data file is not .npy format or has incorrect dimensions.
        """
        # Convert paths to Path objects
        path = Path(path)
        rot_angles_path = Path(rot_angles_path)

        try:
            self.ebsp_dataset = np.load(path)
            logger.info(f"Loaded diffraction pattern data from {path}")
        except Exception as e:
            logger.error(f"Failed to load data from {path}")
            raise ValueError("Only .npy data files are supported.") from e

        if len(self.ebsp_dataset.shape) != 3:
            logger.error(f"Invalid data shape: {self.ebsp_dataset.shape}")
            raise ValueError("The input dataset should be 3D.")

        # Parse rotation angles file more efficiently
        self.rot_angles = self._parse_rotation_angles(rot_angles_path)

        self.transform = transform or create_default_transform(image_size)

        logger.info(f"Dataset initialized with {len(self)} samples")

    def _parse_rotation_angles(self, rot_angles_path: Path) -> pd.DataFrame:
        """Parse rotation angles from the given file.

        Args:
            rot_angles_path: Path to the rotation angles file.

        Returns:
            DataFrame containing rotation angles.

        Raises:
            FileNotFoundError: If the angles file doesn't exist.
            ValueError: If the angles file has incorrect format.
        """
        try:
            with open(rot_angles_path) as f:
                lines = f.readlines()[2:]  # Skip header lines

            angle_list = []
            for line in lines:
                # More efficient parsing using split() and list comprehension
                angles = [angle for angle in line.strip().split(" ") if angle]
                angle_list.append(angles)

            return pd.DataFrame(angle_list, columns=["z1", "x", "z2"]).astype(float)
        except FileNotFoundError:
            logger.error(f"Rotation angles file not found: {rot_angles_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing rotation angles: {e}")
            raise ValueError(f"Failed to parse rotation angles file: {e}") from e

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.ebsp_dataset.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to get.

        Returns:
            Tuple containing (transformed diffraction pattern, rotation angle).
        """
        rot_angle = np.array(self.rot_angles.loc[idx])
        dp = self.ebsp_dataset[idx].astype(np.float64)
        return self.transform(dp), rot_angle


class DPDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Diffraction Pattern data.

    Handles data loading, splitting into train/val/test sets, and creating DataLoaders.

    Attributes:
        dataset_full: Complete dataset containing all samples.
        dataset_train: Training subset of the data.
        dataset_val: Validation subset of the data.
        dataset_test: Test subset of the data (same as full dataset in test mode).
    """

    def __init__(
        self,
        path: str | Path,
        rot_angles_path: str | Path,
        image_size: tuple[int, int] = (128, 128),
        val_data_ratio: float = 0.1,
        batch_size: int = 32,
        n_cpu: int = 4,
        seed: int = 42,
        transform: transforms.Compose | None = None,
    ):
        """Initialize the DataModule.

        Args:
            path: Path to the .npy file containing diffraction patterns.
            rot_angles_path: Path to the file containing rotation angles.
            image_size: Size to crop the images to (height, width).
            val_data_ratio: Fraction of data to use for validation.
            batch_size: Batch size for DataLoaders.
            n_cpu: Number of CPU workers for DataLoaders.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.path = path
        self.rot_angles_path = rot_angles_path
        self.image_size = image_size
        self.val_data_ratio = val_data_ratio
        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.seed = seed
        self.transform = transform or create_default_transform(image_size)

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.dataset_full = DPdataset(
            self.path, self.rot_angles_path, self.image_size, self.transform
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the specified stage.

        Args:
            stage: Either 'fit', 'test', or None.
        """
        if stage == "fit" or stage is None:
            all_size = len(self.dataset_full)
            val_size = int(all_size * self.val_data_ratio)
            train_size = all_size - val_size

            logger.info(
                f"Splitting dataset: {train_size} training, {val_size} validation samples"
            )

            self.dataset_train, self.dataset_val = random_split(
                self.dataset_full,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test":
            self.dataset_test = self.dataset_full
            logger.info(f"Test dataset prepared with {len(self.dataset_test)} samples")

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader.

        Returns:
            DataLoader for training data.
        """
        if self.val_data_ratio > 0.0:
            dataset = self.dataset_train
        else:
            # If no validation split, use the entire dataset for training
            dataset = ConcatDataset([self.dataset_train, self.dataset_val])

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader.

        Returns:
            DataLoader for validation data.
        """
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_cpu,
            pin_memory=True,
            persistent_workers=self.n_cpu > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader.

        Returns:
            DataLoader for test data.
        """
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_cpu,
            pin_memory=True,
            persistent_workers=self.n_cpu > 0,
        )
