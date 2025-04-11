<h1 align="center"> <code>latice</code> Latent-space Autoencoder for Template Indexing of 
Crystallographic EBSD </h1>

<p align="center">
    <img src="assets/latice_logo.png" alt="Latice logo" width="600"/>
</p>

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/latice.git
   cd latice
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   pip install uv
   uv venv .venv --python 3.11

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   uv pip install .
   ```

## Running Tests

The project uses pytest for testing. To run the tests:
```bash
uv run pytest tests
```

## Example Usage

### Training a Model

The primary entry point for training is `train.py`, which uses Hydra for configuration management. The configuration files are located in the `conf` directory.

#### Basic Training

To train the model with default configuration:

```bash
uv run python train.py
```

#### Configuration Overrides
Override configuration parameters directly from the command line:

```bash
uv run python train.py seed=42 data_module.batch_size=128 trainer.max_epochs=100
```


#### Multi-run for Hyperparameter Sweeps
Use Hydra's multi-run functionality:

```bash
uv run python train.py --multirun lightning_module.optimizer_partial.lr=1e-4,5e-4,1e-3 data_module.batch_size=64,128,256
```

### Indexing Diffraction Patterns

The package includes a [Jupyter notebook](notebook/index.ipynb) that demonstrates how to use the LATICE system for indexing electron backscatter diffraction (EBSD) patterns: 

```python
from latice.index.dp_indexer import DiffractionPatternIndexer, IndexerConfig
from latice.index.chroma_db import LatentVectorDatabaseConfig, LatentVectorDatabase
from latice.model import VariationalAutoEncoderRawData
import torch
from pathlib import Path

# Set up the indexer
db = LatentVectorDatabase(
    config=LatentVectorDatabaseConfig(
        collection_name="my_collection",
        persist_directory=".chroma_db",
        dimension=16,
    )
)

# Load model
model = VariationalAutoEncoderRawData()
checkpoint = torch.load("path/to/checkpoints/vae-best.pt", map_location="cuda")
model.load_state_dict(checkpoint)
model.eval()

# Initialise indexer to get top-10 simarity search
indexer = DiffractionPatternIndexer(
    model=model,
    db=db,
    config=IndexerConfig(
        pattern_path=Path("path/to/patterns.npy"),
        angles_path=Path("path/to/angles.txt"),
        batch_size=64,
        image_size=(128, 128),
        top_n=10,
        orientation_threshold=3.0,
    ),
)

# Build dictionary from patterns
indexer.build_dictionary()

# Index a single pattern
pattern = load_your_pattern()  # Load your pattern as numpy array
result = indexer.index_pattern(pattern)
print(f"Indexed orientation: {result.mean_orientation}")
print(f"Success: {result.success}")
```

For detailed examples and visualization, refer to the [notebook/index.ipynb](notebook/index.ipynb).