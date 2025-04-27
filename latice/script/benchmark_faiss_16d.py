import fire
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import logging

import pandas as pd
import torch
from latice.index.faiss_db import (
    FaissLatentVectorDatabase,
    FaissLatentVectorDatabaseConfig,
)
from latice.index.dp_indexer import DiffractionPatternIndexer, IndexerConfig
import time
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)

from latice.model import VariationalAutoEncoderRawData

logger = logging.getLogger(__name__)


def time_indexing(
    batch_patterns: NDArray[np.float32],
    indexer: DiffractionPatternIndexer,
    n_samples: int = 1000,
    top_n: int = 1,
    orientation_threshold: float = 3.0,
) -> list[float]:
    """Time the indexing process."""
    num_rows = len(batch_patterns)
    random_indices = np.random.choice(num_rows, size=n_samples, replace=False)
    all_index_times = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Indexing patterns...", total=n_samples)
        for i in random_indices:
            start_time = time.time()
            _ = indexer.index_pattern(
                batch_patterns[i],
                top_n=top_n,
                orientation_threshold=orientation_threshold,
            )
            end_time = time.time()
            index_time = end_time - start_time
            all_index_times.append(index_time)
            progress.update(task, advance=1, time_elapsed=index_time)
    return all_index_times


def main(
    pattern_path: Path,
    angles_path: Path,
    npz_path: Path = "faiss_index.npz",
    dimension: int = 16,
    device: str = "cpu",
    model_path: str = "checkpoints/vae-best.pt",
    batch_size: int = 32,
    random_seed: int = 42,
    image_size: tuple[int, int] = (128, 128),
    top_n: int = 1,
    orientation_threshold: float = 3.0,
    n_samples: int = 1000,
    output_path: Path = "benchmark_faiss_16d.csv",
):
    batch_patterns = np.load(pattern_path, mmap_mode="r")
    faiss_db = FaissLatentVectorDatabase(
        config=FaissLatentVectorDatabaseConfig(npz_path=npz_path, dimension=dimension)
    )

    model = VariationalAutoEncoderRawData()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    latent_indexer = DiffractionPatternIndexer(
        model=model,
        db=faiss_db,
        config=IndexerConfig(
            pattern_path=pattern_path,
            angles_path=angles_path,
            batch_size=batch_size,
            device=device,
            latent_dim=dimension,
            random_seed=random_seed,
            image_size=image_size,
            top_n=top_n,
            orientation_threshold=orientation_threshold,
        ),
    )
    latent_indexer.build_dictionary()

    all_index_times = time_indexing(
        batch_patterns=batch_patterns, indexer=latent_indexer, n_samples=n_samples
    )

    df_times = pd.DataFrame(all_index_times, columns=["index_time"])
    df_times.to_csv(output_path, index=False)
    logger.info(f"Times saved to {output_path}")
    logger.info(f"Mean time per pattern: {np.mean(all_index_times):.4f} seconds")
    logger.info(
        f"Standard deviation of time per pattern: {np.std(all_index_times):.4f} seconds"
    )


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()
