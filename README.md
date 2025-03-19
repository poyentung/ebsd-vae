# Learning crystallagrpic orientation from EBSD patterns



## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ebsd-vae.git
   cd diffraction-pattern-vae
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   pip install uv
   uv venv .venv python=3.11

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   uv pip install .
   ```

## Example Usage

### Training a Model

The primary entry point for training is `train.py`, which uses Hydra for configuration management. The configuration files are located in the `conf` directory.

#### Basic Training

To train the model with default configuration:

```bash
python train.py
```

#### Configuration Overrides
Override configuration parameters directly from the command line:

```bash
python train.py seed=42 data_module.batch_size=128 trainer.max_epochs=100
```


#### Multi-run for Hyperparameter Sweeps
Use Hydra's multi-run functionality:

```bash
python train.py --multirun lightning_module.optimizer_partial.lr=1e-4,5e-4,1e-3 data_module.batch_size=64,128,256
```
