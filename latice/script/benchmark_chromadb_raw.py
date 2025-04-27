import fire
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import logging

import pandas as pd
from latice.index.chroma_db import (
    ChromaLatentVectorDatabaseConfig,
    ChromaLatentVectorDatabase,
)
from latice.index.raw_dp_indexer import RawDiffractionPatternIndexer, RawIndexerConfig
import time
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)

logger = logging.getLogger(__name__)


def time_indexing(
    batch_patterns: NDArray[np.float32],
    raw_indexer: RawDiffractionPatternIndexer,
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
            _ = raw_indexer.index_pattern(
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
    batch_size: int = 32,
    random_seed: int = 42,
    image_size: tuple[int, int] = (128, 128),
    top_n: int = 1,
    orientation_threshold: float = 3.0,
    n_samples: int = 1000,
    persist_directory: str = "notebook/.chroma_db_raw",
    output_path: Path = "benchmark_chromadb_raw.csv",
):
    batch_patterns = np.load(pattern_path, mmap_mode="r")
    raw_dimension = image_size[0] * image_size[1]
    logger.info(f"Raw pattern dimension: {raw_dimension}")

    chroma_raw_db = ChromaLatentVectorDatabase(
        config=ChromaLatentVectorDatabaseConfig(
            persist_directory=persist_directory, dimension=raw_dimension
        )
    )
    raw_indexer = RawDiffractionPatternIndexer(
        config=RawIndexerConfig(
            pattern_path=pattern_path,
            angles_path=angles_path,
            batch_size=batch_size,
            random_seed=random_seed,
            image_size=image_size,
            top_n=top_n,
            orientation_threshold=orientation_threshold,
        ),
        db=chroma_raw_db,
    )

    all_index_times = time_indexing(
        batch_patterns=batch_patterns, raw_indexer=raw_indexer, n_samples=n_samples
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
