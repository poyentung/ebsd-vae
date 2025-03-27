import torch
import random
import numpy as np
from numpy.typing import NDArray
from math import pi, sqrt
import altair as alt
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from torch.utils.data import Dataset
from matplotlib.figure import Figure
from PIL import Image

from scipy.spatial.transform import Rotation as R
from src.utils.colorkey import ColorKeyGenerator

# Constants
PI_OVER_180 = pi / 180
K_180_OVER_PI = 180 / pi
SQRT2_INV = 1 / sqrt(2)
SQRT3_INV = 1 / sqrt(3)
USE_INVERSION = True

CUBIC_SYMMETRY = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5, -0.5],
    [0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5, -0.5],
    [SQRT2_INV, SQRT2_INV, 0, 0],
    [SQRT2_INV, 0, SQRT2_INV, 0],
    [SQRT2_INV, 0, 0, SQRT2_INV],
    [SQRT2_INV, -SQRT2_INV, 0, 0],
    [SQRT2_INV, 0, -SQRT2_INV, 0],
    [SQRT2_INV, 0, 0, -SQRT2_INV],
    [0, SQRT2_INV, SQRT2_INV, 0],
    [0, -SQRT2_INV, SQRT2_INV, 0],
    [0, 0, SQRT2_INV, SQRT2_INV],
    [0, 0, -SQRT2_INV, SQRT2_INV],
    [0, SQRT2_INV, 0, SQRT2_INV],
    [0, -SQRT2_INV, 0, SQRT2_INV],
]
QUAT_SYM = R.from_quat(CUBIC_SYMMETRY)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def plot_detection(
    imgs: torch.Tensor,
    recon_imgs: torch.Tensor,
    cmap: str = "viridis",
    num_samples: int = 4,
    figsize: tuple[int, int] = (10, 5),
    dpi: int = 150,
) -> Figure:
    """Plot original and reconstructed images for visual comparison.

    Args:
        imgs: Input images tensor.
        recon_imgs: Reconstructed images tensor.
        cmap: Colormap to use for plotting.
        num_samples: Number of random samples to display.
        figsize: Figure dimensions (width, height) in inches.
        dpi: Figure resolution.

    Returns:
        matplotlib Figure object containing the comparison plots.
    """
    imgs = imgs.to("cpu")
    recon_imgs = torch.sigmoid(recon_imgs)
    recon_imgs = recon_imgs.to("cpu")

    img_ids = random.sample(range(imgs.size(0)), num_samples)
    fig, axs = plt.subplots(2, num_samples, figsize=figsize, dpi=dpi)

    for i in range(2):
        for j in range(num_samples):
            img_id = img_ids[j]
            if i == 0:
                axs[i, j].imshow(imgs[img_id].squeeze().numpy(), cmap=cmap)
            else:
                axs[i, j].imshow(recon_imgs[img_id].squeeze().numpy(), cmap=cmap)
            axs[i, j].axis("off")

    fig.subplots_adjust(wspace=0.0, hspace=0.05)
    return fig


def log_fig(
    log_name: str,
    fig: Figure,
    logger: pl.loggers.TensorBoardLogger | pl.loggers.WandbLogger,
    current_epoch: int,
) -> None:
    """Log a matplotlib figure to a PyTorch Lightning logger.

    Args:
        log_name: Name to use for the logged image.
        fig: Figure to log.
        logger: PyTorch Lightning logger (TensorBoard or Weights & Biases).
        current_epoch: Current training epoch.

    Returns:
        None
    """
    fig.canvas.draw()

    # Convert figure to numpy array
    eval_result = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    # Handle different logger types
    if isinstance(logger, pl.loggers.TensorBoardLogger):
        eval_result = np.moveaxis(eval_result[:, :, :3], 2, 0)
        logger.experiment.add_image(f"{log_name}_{current_epoch}", eval_result)
    elif isinstance(logger, pl.loggers.WandbLogger):
        eval_result = Image.fromarray(eval_result[:, :, :3])
        logger.log_image(key=f"{log_name}_{current_epoch}", images=[eval_result])


def plot_latent(
    dataset: Dataset, latent: np.ndarray, color: str = "ipf_z"
) -> alt.Chart:
    """Plot 2D latent space visualization with color coding.

    Args:
        dataset: Dataset containing rotation angles.
        latent: 2D latent space vectors to plot.
        color: Color coding method ('ipf_x', 'ipf_y', or 'ipf_z').

    Returns:
        Altair chart object displaying the latent space.
    """
    source = dataset.rot_angles.copy()
    if color in ("ipf_x", "ipf_y", "ipf_z"):
        source["color"] = get_color_key(source.to_numpy(), mode=color, hex_string=True)
    source["latent_x"] = latent[:, 0]
    source["latent_y"] = latent[:, 1]

    # Enable handling large datasets
    alt.data_transformers.disable_max_rows()

    # Create interactive selection
    interaction = alt.selection(
        type="interval",
        bind="scales",
        on="[mousedown[event.shiftKey], mouseup] > mousemove",
        translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
        zoom="wheel![event.shiftKey]",
    )

    # Create main points visualization
    points = (
        alt.Chart(source)
        .mark_circle(size=20.0, color="red")
        .encode(
            x="latent_x:Q",
            y="latent_y:Q",
            color=alt.Color("color", scale=None),
            tooltip=[
                alt.Tooltip("latent_x:Q", format=",.2f"),
                alt.Tooltip("latent_y:Q", format=",.2f"),
                alt.Tooltip("z1:Q", format=",.2f"),
                alt.Tooltip("x:Q", format=",.2f"),
                alt.Tooltip("z2:Q", format=",.2f"),
            ],
        )
        .properties(width=450, height=450)
        .properties(title=alt.TitleParams(text="Latent space"))
        .add_selection(interaction)
    )

    return points


def get_color_key(
    rot_angle: NDArray, mode: str = "ipf_z", hex_string: bool = False
) -> NDArray | list[str]:
    """Generate color keys for rotation angles.

    Args:
        rot_angle: Array of rotation angles.
        mode: Projection mode, one of 'ipf_x', 'ipf_y', 'ipf_z'.
        hex_string: Whether to return colors as hex strings.

    Returns:
        Array of RGB colors or list of hex color strings.
    """
    # Handle single rotation angle
    rot_angle = rot_angle[np.newaxis, :] if len(rot_angle.shape) < 2 else rot_angle

    pole = R.from_euler("zxz", rot_angle, degrees=True).as_matrix()

    if mode == "ipf_z":
        pole = pole[:, 2, :]
    elif mode == "ipf_y":
        pole = pole[:, 1, :]
    elif mode == "ipf_x":
        pole = pole[:, 0, :]

    ckey_generator = ColorKeyGenerator()

    color_keys = []
    for p in pole:
        color_keys.append(ckey_generator.generate_ipf_color(zone_axis=p))

    if not hex_string:
        return np.array(color_keys)
    else:
        return ["#{:02x}{:02x}{:02x}".format(*rgb) for rgb in color_keys]
