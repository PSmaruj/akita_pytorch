"""
Akita v2 model architecture for genome folding prediction.

This module implements the complete Akita v2 neural network architecture,
which predicts Hi-C contact matrices from DNA sequences.
"""

import torch
import torch.nn as nn

from akita_model.modules import (
    StochasticReverseComplement, 
    StochasticShift, 
    ConvBlock, 
    ConvTower,
    ResidualDilatedBlock1D, 
    ConvBlockReduce, 
    OneToTwo,
    SqueezeExcite,
    Conv2DBlock,
    Symmetrize2D,
    DilatedResidualBlock2D,
    Cropping2D,
    UpperTri,
    Final
)


class SeqNN(nn.Module):
    """
    Akita v2 sequence-to-contact neural network.
    
    This model predicts Hi-C contact matrices from DNA sequences using a
    two-stage architecture:
    1. Trunk: 1D convolutions on the sequence to extract features
    2. Head: 2D convolutions on pairwise features to predict contacts
    
    Args:
        n_channel (int): Number of input channels (4 for one-hot DNA). Default: 4
        n_targets (int): Number of output targets/tracks. Default: 1
    
    Input shape:
        (batch_size, 4, 524288) - One-hot encoded DNA sequences
    
    Output shape:
        (batch_size, n_targets, num_upper_triangular_elements)
    
    Architecture overview:
        - Data augmentation: Stochastic reverse complement & shift
        - Initial conv block + conv tower (10 layers)
        - 11 residual dilated blocks with increasing dilation (Fibonacci-like)
        - Channel reduction (128 -> 80)
        - 1D to 2D transformation
        - 6 residual dilated 2D blocks
        - Squeeze-and-excite attention
        - Spatial cropping
        - Upper triangular extraction
        - Final linear projection
    """
    
    def __init__(self, n_channel=4, n_targets=5):
        super(SeqNN, self).__init__()

        # ====================================================================
        # TRUNK - 1D sequence processing
        # ====================================================================
        
        # Data augmentation layers (training only)
        self.stochastic_reverse_complement = StochasticReverseComplement()
        self.stochastic_shift = StochasticShift(
            shift_max=11, 
            symmetric=True, 
            pad='constant'
        )

        # Initial convolution block
        self.conv_block_1 = ConvBlock(
            in_channels=n_channel,
            filters=128,
            kernel_size=15,
            stride=1,
            dilation_rate=1,
            pool_size=2,
            pool_type='max',
            norm_type='batch',
            bn_momentum=0.1,
            use_dropout=False
        )

        # Convolutional tower (10 repeated conv+pool blocks)
        self.conv_tower = ConvTower(
            in_channels=128,
            filters_init=128,
            filters_mult=1.0,
            kernel_size=5,
            pool_size=2,
            repeat=10,
            norm_type='batch',
            bn_momentum=0.1
        )

        # Dilated residual blocks with Fibonacci-like dilation rates
        # These capture multi-scale features with increasing receptive fields
        dilation_rates = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 145]
        self.residual1d_blocks = nn.ModuleList([
            ResidualDilatedBlock1D(
                in_channels=128,
                mid_channels=64,
                dropout_rate=0.1,
                dilation_rate=rate,
                bn_momentum=0.1,
                norm_type='batch'
            ) for rate in dilation_rates
        ])
        
        # Channel reduction layer
        self.conv_reduce = ConvBlockReduce(
            in_channels=128,
            out_channels=80,
            kernel_size=5,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        # ====================================================================
        # HEAD - 2D contact matrix prediction
        # ====================================================================
        
        # Transform 1D features to 2D pairwise features
        self.one_to_two = OneToTwo(operation='mean')
        
        # Initial 2D convolution
        self.conv2d_block = Conv2DBlock(
            in_channels=80, 
            out_channels=80, 
            kernel_size=3,
            norm_type='batch'
        )
        
        # Enforce matrix symmetry
        self.symmetrize_2d = Symmetrize2D()
        
        # 2D dilated residual blocks
        dilation_rates_2d = [1, 2, 4, 7, 12, 21]
        self.residual2d_blocks = nn.ModuleList([
            DilatedResidualBlock2D(
                in_channels=80, 
                mid_channels=40, 
                kernel_size=3, 
                dilation_rate=rate,
                dropout_prob=0.1,
                norm_type='batch'
            ) for rate in dilation_rates_2d
        ])
        
        # Channel attention mechanism
        self.squeeze_excite = SqueezeExcite(
            in_channels=80,
            activation='relu',
            additive=True,
            bottleneck_ratio=8,
            norm_type='batch',
            bn_momentum=0.9
        )
        
        # Spatial cropping to remove boundary artifacts
        self.cropping_2d = Cropping2D(cropping=64)
        
        # Extract upper triangular portion (matrices are symmetric)
        self.upper_tri = UpperTri(diagonal_offset=2)
        
        # Final projection to target dimension
        self.final = Final(units=1, activation='linear')
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input DNA sequences (batch_size, 4, sequence_length)
        
        Returns:
            Tensor: Predicted Hi-C contacts (batch_size, n_targets, num_contacts)
        """
        device = x.device
        self.to(device)

        # ====================================================================
        # Data augmentation (training only)
        # ====================================================================
        x, reverse_bool = self.stochastic_reverse_complement(x)
        x = self.stochastic_shift(x)

        # ====================================================================
        # TRUNK - 1D processing
        # ====================================================================
        x = self.conv_block_1(x)
        x = self.conv_tower(x)
        
        # Apply all 1D residual blocks
        for residual_block in self.residual1d_blocks:
            x = residual_block(x)
        
        x = self.conv_reduce(x)
        
        # ====================================================================
        # HEAD - 2D processing
        # ====================================================================
        x = self.one_to_two(x)
        x = self.conv2d_block(x)
        x = self.symmetrize_2d(x)
        
        # Apply all 2D residual blocks
        for residual_block in self.residual2d_blocks:
            x = residual_block(x)
        
        x = self.squeeze_excite(x)
        x = self.cropping_2d(x)
        
        # ====================================================================
        # Output processing
        # ====================================================================
        x = self.upper_tri(x, reverse_bool)        
        x = self.final(x)
        
        return x
