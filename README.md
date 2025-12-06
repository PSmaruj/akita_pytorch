# PyTorch AkitaV2

A PyTorch implementation of the AkitaV2 deep learning model for predicting Hi-C contact matrices from DNA sequences.

## Overview

This repository provides a PyTorch version of the [AkitaV2 model](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012824), which is an improved version of the original [Akita model](https://www.nature.com/articles/s41592-020-0958-x). AkitaV2 predicts 3D genome organization (Hi-C contact maps) directly from DNA sequence using deep convolutional neural networks.

### Model Transfer and Fine-tuning

The models in this repository were transferred from the original TensorFlow implementation to PyTorch. The TensorFlow AkitaV2 model was trained jointly on both mouse and human data. Our transferred PyTorch models are introduced in this preprint [PLACEHOLDER FOR LINK].

**Important Note on Performance**: The PyTorch implementation showed a minor performance reduction relative to the TensorFlow model, likely due to small numerical and implementation differences between frameworks (similar issues were reported in the [Basenji2 PyTorch port](https://github.com/d-laub/basenji2-pytorch/tree/main)). To mitigate this effect, each PyTorch model was fine-tuned from the transferred TensorFlow weights for the corresponding cell type. As a result, we provide separate fine-tuned models for each cell type/dataset.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pytorch_akita.git
cd pytorch_akita
```

### 2. Set Up Environment

#### Option A: Using Conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate pytorch_akita
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Run tests to verify installation
pytest tests/ -v
```

## Repository Structure

```
pytorch_akita/
├── akita_model/                # Core model architecture
│   ├── model.py                # Main Akita v2 model (clean implementation)
│   ├── model_v2_compatible.py  # Compatible with old checkpoints
│   └── modules.py              # Neural network building blocks
│
├── data_processing/            # Data preprocessing utilities
│   ├── dataset.py              # PyTorch Dataset for Hi-C data
│   ├── preprocessing_data_parallel.py  # Parallel data preprocessing
│   └── visualize_training_data.ipynb   # Data visualization notebook
│
├── training/                   # Training from scratch
│   ├── training_utils.py       # Shared training utilities
│   ├── train_model.py          # Training script
│   └── train_model.sh          # SLURM job script
│
├── finetuning/                 # Fine-tuning pretrained models
│   ├── finetune_model.py       # Fine-tuning script
│   ├── finetune_model.sh       # SLURM job script
│   └── analyze_finetuning_loss.ipynb  # Loss analysis notebook
│
├── evaluation/                 # Model evaluation
│   └── evaluate_model.ipynb    # Evaluation and visualization notebook
│
├── weight_transfer/            # TensorFlow to PyTorch conversion
│   └── transfer_tf_to_torch.py # Weight transfer script
│
├── utils/                      # Shared utility functions
│   ├── visualization_utils.py  # Plotting and visualization
│   └── analysis_utils.py       # Loss analysis and metrics
│
├── tests/                      # Unit tests
│   └── test_utils.py           # Tests for critical functions
│
├── models/                     # Pretrained and fine-tuned models
│   ├── tf_transferred/         # Models transferred from TensorFlow
│   │   ├── human_models/
│   │   └── mouse_models/
│   └── finetuned/              # Fine-tuned models (recommended)
│       ├── human_models/
│       └── mouse_models/
│
├── environment.yml             # Conda environment file
├── requirements.txt            # pip requirements file
├── README.md                   # This file
└── LICENSE                     # MIT License
```

## Available Models

Pretrained models are available in the `models/` directory:

- **`models/tf_transferred/`**: Models with weights transferred from TensorFlow
- **`models/finetuned/`**: Fine-tuned models for specific cell types

### Organisms and Cell Types

Information on datasets used for training can be found in the supporting information table in the Akita v2 paper: [PLOS Computational Biology Supporting Information](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012824#sec030).

**Mouse B cell data**: We collected Hi-C data from B cells via the [4DN data portal](https://data.4dnucleome.org/) ([Vian et al., 2018](https://www.sciencedirect.com/science/article/pii/S0092867418304045)), specifically the paired files `4DNFI27I3P1V` and `4DNFIFBBAKK4`, which were concatenated to increase coverage.

## Usage

### Loading a Pretrained Model

```python
import torch
from akita_model.model import SeqNN

# Load a fine-tuned model
model_path = "models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

print("Model loaded successfully!")
```

### Making Predictions

```python
import torch
import numpy as np

# Prepare input: one-hot encoded DNA sequence
# Shape: (batch_size, 4, 1310720)
# Channels: [A, C, G, T]
sequence = torch.randn(1, 4, 1310720)  # Replace with actual one-hot encoded sequence

# Move to device
sequence = sequence.to(device)
model = model.to(device)

# Make prediction
with torch.no_grad():
    prediction = model(sequence)

# Output shape: (batch_size, n_targets, num_contacts)
print(f"Prediction shape: {prediction.shape}")
```

### Fine-tuning on Your Own Data

See `finetuning/finetune_model.py` and `finetuning/finetune_model.sh` for examples of fine-tuning the model on your own Hi-C datasets.

```bash
python finetuning/finetune_model.py \
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
- **`finetuning/analyze_finetuning_loss.ipynb`**: Analyze training/validation loss curves
- **`evaluation/evaluate_model.ipynb`**: Evaluate model performance and visualize predictions

## Testing

Run the test suite to verify correctness of critical functions:

```bash
# Run all tests
pytest tests/ -v
```

## Citation

If you use this code in your research, please cite:

**Sequence Design for Genome Folding using PyTorch Akita**:
```
[PLACEHOLDER - Add your preprint citation]
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

If you encounter any bugs or have feature requests, please [open an issue](https://github.com/yourusername/pytorch_akita/issues) with:
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

**Last Updated**: December 2025