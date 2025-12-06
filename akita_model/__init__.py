"""
Akita v2 model architecture for genome folding prediction.

This package contains the PyTorch implementation of the Akita v2 model
for predicting Hi-C contact matrices from DNA sequences.
"""

from .model import SeqNN

__version__ = '0.1.0'      # package version (pre-release)
__akita_version__ = 'v2'   # Akita architecture version
__all__ = ['SeqNN']
