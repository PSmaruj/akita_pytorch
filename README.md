# PyTorch AkitaV2

A PyTorch implementation of the AkitaV2 deep learning model for predicting Hi-C contact matrices from DNA sequences.

## Overview

This repository provides a PyTorch version of the [AkitaV2 model](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012824), which is an improved version of the original [Akita model](https://www.nature.com/articles/s41592-020-0958-x). AkitaV2 predicts 3D genome organization (Hi-C contact maps) directly from DNA sequence using deep convolutional neural networks.

### Model Transfer and Fine-tuning

The models in this repository were transferred from the original TensorFlow implementation to PyTorch. The TensorFlow AkitaV2 model was trained jointly on both mouse and human data. Our transferred PyTorch models are introduced in this preprint [PLACEHOLDER FOR LINK].

**Motivation for fine-tuning**: The PyTorch implementation showed a minor performance reduction relative to the TensorFlow model after directly transferring weights, likely due to small numerical and implementation differences between frameworks (similar issues were reported in the [Basenji2 PyTorch port](https://github.com/d-laub/basenji2-pytorch/tree/main)). To mitigate this effect, each PyTorch model was fine-tuned from the transferred TensorFlow weights for the corresponding cell type. As a result, we provide separate fine-tuned models for each cell type/dataset.

## Installation

### Option A — pip (recommended for most users)

If you want to use the model class to train on your own data:

```bash
pip install git+https://github.com/PSmaruj/akita_pytorch.git
```

> **Note:** The pip installation includes the model class and training code,
> but **not the model weights**. If you plan to run inference or use AkitaSF,
> you will need the pretrained checkpoints — see [Downloading Model Weights](#downloading-model-weights) below.

### Option B — clone (required for running workflows)

```bash
# Clone the repository
git clone https://github.com/PSmaruj/akita_pytorch.git
cd akita_pytorch

# Create and activate the conda environment
conda env create -f environment.yml
conda activate pytorch_akita
```

### Verify Installation

```bash
pytest tests/ -v
```

## Repository Structure

```
akita_pytorch/
├── 🧬 akita_model/           # Core architecture & neural building blocks
│   ├── model.py              # Main Akita V2 implementation
│   └── modules.py            # Custom layers and modules
│
├── 📊 data_processing/       # Hi-C data pipelines & preprocessing
│   ├── dataset.py            # PyTorch Dataset for HiC data
│   ├── create_cool_files.sh  # .pairs to .cool conversion
│   ├── preprocessing_...     # Parallel processing scripts (.py/.sh)
│   └── visualization_...     # Notebook for data sanity checks
│
├── 🚀 workflows/             # Training, Finetuning, & Weights Transfer
│   ├── training/             # Scripts & SLURM jobs for scratch training
│   ├── finetuning/           # Pretrained model adaptation & loss analysis
│   └── weight_transfer/      # TF-to-PyTorch conversion logic
│
├── 🧪 evaluation/            # The Performance Hub
│   ├── internal/             # Notebooks/scripts for average model performance
│   └── benchmarking/         # Cross-model evaluation (ORCA, AlphaGenome)
│
├── 🛠️ utils/                 # Shared helper functions
│   ├── analysis_utils.py     # Metrics and stats
│   ├── data_utils.py         # I/O helpers
│   └── visualization_utils.py # Contact map plotting functions
│
├── 📋 tests/                 # Unit tests for critical functions
├── 🖼️ plots/                 # Figures for the paper
└── 📄 Config & Meta          # environment.yml, requirements.txt, LICENSE
```

## Downloading Model Weights

Model weights are hosted on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19599537.svg)](https://doi.org/10.5281/zenodo.19599537)

Two sets of weights are available:

- **`finetuned/`** — Recommended for inference; fine-tuned per cell type/dataset
- **`trained_from_scratch/`** — Weights from native PyTorch training on B-cell data

To download and extract all weights, run from the root of the repository:

```bash
wget -O akita_pytorch_model_weights.tar.gz https://zenodo.org/record/19599537/files/akita_pytorch_model_weights.tar.gz
tar -xzf akita_pytorch_model_weights.tar.gz && rm akita_pytorch_model_weights.tar.gz
```

This will recreate the `models/` directory with the original structure:

```
models/
├── finetuned/
│   ├── mouse/
│   │   └── <dataset>/checkpoints/*.pth
│   └── human/
│       └── <dataset>/checkpoints/*.pth
└── trained_from_scratch/
```

## Making Predictions

```python
import torch
from akita_model.model import SeqNN

# Load a fine-tuned model (after downloading the models)
model_path = "models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Prepare input: one-hot encoded DNA sequence
# Shape: (batch_size, 4, 1310720) — channels: [A, C, G, T]
sequence = torch.randn(1, 4, 1310720)  # Replace with actual one-hot encoded sequence
sequence = sequence.to(device)

with torch.no_grad():
    prediction = model(sequence)

# Output shape: (batch_size, n_targets, num_contacts)
print(f"Prediction shape: {prediction.shape}")
```

## Fine-tuning on Your Own Data

See `workflows/finetuning/finetune_model.py` and `workflows/finetuning/finetune_model.sh` for examples of fine-tuning the model on your own Hi-C datasets.

```bash
python workflows/finetuning/finetune_model.py \
    --data_dir /path/to/data \
    --test_fold fold0 \
    --val_fold fold1 \
    --data_name MyDataset \
    --organism mouse \
    --data-split 0 \
    --epochs 70 \
    --lr 0.001 \
    --save-model
```

## Data Preprocessing

To preprocess your own Hi-C data:

1. **Create cooler files** from Hi-C pairs files:
   ```bash
   bash data_processing/create_cool_files.sh
   ```

2. **Preprocess into PyTorch tensors**:
   ```bash
   bash data_processing/preprocessing_data_parallel.sh
   ```

See the scripts for detailed configuration options.

## Evaluation

Use the provided Jupyter notebooks for evaluation and visualization:

- **`data_processing/visualize_training_data.ipynb`**: Visualize training data quality
- **`workflows/finetuning/analyze_finetuning_loss.ipynb`**: Analyze training/validation loss curves
- **`evaluation/internal/evaluate_model.ipynb`**: Evaluate model performance and visualize predictions

## Testing

Run the test suite to verify correctness of critical functions:

```bash
pytest tests/ -v
```

## Citation

If you use this code in your research, please cite:

**Sequence Design for Genome Folding using PyTorch Akita**:
```
[PLACEHOLDER - our preprint citation]
```

**AkitaV2 (original TensorFlow implementation)**:
```
Smaruj PN, Kamulegeya F, Kelley DR, Fudenberg G (2025) 
Interpreting the CTCF-mediated sequence grammar of genome folding with Akita v2. 
PLOS Computational Biology 21(2): e1012824. 
https://doi.org/10.1371/journal.pcbi.1012824
```

## Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

For major changes, please open an issue first to discuss what you would like to change.

### Reporting Issues

If you encounter any bugs or have feature requests, please [open an issue](https://github.com/PSmaruj/akita_pytorch/issues) with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment details (OS, Python version, PyTorch version)

## Contact Information

Feedback and questions are appreciated. Please contact us at:

- **Geoffrey Fudenberg**: fudenber at usc dot edu
- **Paulina Smaruj**: smaruj at usc dot edu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Resources

- [Original Akita (Nature Methods)](https://www.nature.com/articles/s41592-020-0958-x)
- [AkitaV2 (PLOS Computational Biology)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012824)
- [Basenji Repository](https://github.com/calico/basenji)
- [Cooler: Hi-C data format](https://cooler.readthedocs.io/)
- [Cooltools](https://cooltools.readthedocs.io/en/latest/)

---

**Last Updated**: April 2026