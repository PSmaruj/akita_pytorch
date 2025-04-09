import numpy as np
import random

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


from sklearn.metrics import r2_score
from modules import (StochasticReverseComplement, 
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
                     Final)


def checkpointed(module, x):
    def custom_forward(*inputs):
        return module(*inputs)
    return checkpoint(custom_forward, x, use_reentrant=False)


class SeqNN(nn.Module):
    def __init__(self, n_channel=4):
        super(SeqNN, self).__init__()

        # TRUNK
        self.stochastic_reverse_complement = StochasticReverseComplement()
        self.stochastic_shift = StochasticShift(shift_max=11, symmetric=True, pad='constant')

        # ConvBlock
        self.conv_block_1 = ConvBlock(
            in_channels=n_channel,
            filters=128,
            kernel_size=15,
            stride=1,
            dilation_rate=1,
            pool_size=2,
            pool_type='max',
            norm_type='batch',
            bn_momentum=0.1, # 1 - tensorflow momentum
            use_dropout=False
        )

        # ConvTower
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

        # ResidualDilatedBlock
        self.residual1d_block1 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=1,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block2 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=2,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block3 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=3,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block4 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=5,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block5 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=8,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block6 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=13,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block7 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=21,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block8 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=34,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block9 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=55,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block10 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=89,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        self.residual1d_block11 = ResidualDilatedBlock1D(
            in_channels=128,
            mid_channels=64,
            dropout_rate=0.1,
            dilation_rate=145,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        # ConvBlockReduce
        self.conv_reduce = ConvBlockReduce(
            in_channels=128,
            out_channels=80,
            kernel_size=5,
            bn_momentum=0.1,
            norm_type='batch'
        )
        
        # HEAD
        
        # 1D to 2D
        self.one_to_two = OneToTwo(operation='mean')
        
        # Concatenating distance - no in V2
        # self.concat_dist = ConcatDist2D()
        
        # ConvBlock
        self.conv2d_block = Conv2DBlock(
            in_channels=80, 
            out_channels=80, 
            kernel_size=3,
            norm_type='batch')
        
        # Symmetrize2D
        self.symmetrize_2d = Symmetrize2D()
        
        # ResidualDilatedBlock - 2D
        self.residual2d_block1 = DilatedResidualBlock2D(
            in_channels=80, 
            mid_channels=40, 
            kernel_size=3, 
            dilation_rate=1,
            dropout_prob=0.1,
            norm_type='batch')
        
        self.residual2d_block2 = DilatedResidualBlock2D(
            in_channels=80, 
            mid_channels=40, 
            kernel_size=3, 
            dilation_rate=2,
            dropout_prob=0.1,
            norm_type='batch')
        
        self.residual2d_block3 = DilatedResidualBlock2D(
            in_channels=80, 
            mid_channels=40, 
            kernel_size=3, 
            dilation_rate=4,
            dropout_prob=0.1,
            norm_type='batch')
        
        self.residual2d_block4 = DilatedResidualBlock2D(
            in_channels=80, 
            mid_channels=40, 
            kernel_size=3, 
            dilation_rate=7,
            dropout_prob=0.1,
            norm_type='batch')
        
        self.residual2d_block5 = DilatedResidualBlock2D(
            in_channels=80, 
            mid_channels=40, 
            kernel_size=3, 
            dilation_rate=12,
            dropout_prob=0.1,
            norm_type='batch')
        
        self.residual2d_block6 = DilatedResidualBlock2D(
            in_channels=80, 
            mid_channels=40, 
            kernel_size=3, 
            dilation_rate=21,
            dropout_prob=0.1,
            norm_type='batch')
        
        
        self.squeeze_excite = SqueezeExcite(
            in_channels=80,  # Should match the output channels of `residual2d_block6`
            activation='relu',
            additive=True,
            bottleneck_ratio=8,
            norm_type='batch',  # Same normalization type as your model
            bn_momentum=0.9
        )
        
        # Cropping2D
        # 64 for V2
        self.cropping_2d = Cropping2D(cropping=64)
        
        # UpperTri
        self.upper_tri = UpperTri(diagonal_offset=2)
        
        # Final
        # V2 6 for mouse, 5 for human
        self.final = Final(units=1, activation='linear')
        
    def forward(self, x):
        device = x.device
        self.to(device)

        # Apply stochastic reverse complement
        x, reverse_bool = self.stochastic_reverse_complement(x)

        # Apply stochastic shift
        x = self.stochastic_shift(x)

        # Apply ConvBlock
        x = checkpointed(self.conv_block_1, x)
        
        # Apply ConvTower
        x = checkpointed(self.conv_tower, x)
        
        # Apply ResidualDilatedBlock - 1D
        x = checkpointed(self.residual1d_block1, x)
        x = checkpointed(self.residual1d_block2, x)
        x = checkpointed(self.residual1d_block3, x)
        x = checkpointed(self.residual1d_block4, x)
        x = checkpointed(self.residual1d_block5, x)
        x = checkpointed(self.residual1d_block6, x)
        x = checkpointed(self.residual1d_block7, x)
        x = checkpointed(self.residual1d_block8, x)
        x = checkpointed(self.residual1d_block9, x)
        x = checkpointed(self.residual1d_block10, x)
        x = checkpointed(self.residual1d_block11, x)
        
        # Apply ConvBlockReduce
        x = self.conv_reduce(x)
        
        # Apply OneToTwo
        x = self.one_to_two(x)
        
        # Apply ConcatDist2D
        # x = self.concat_dist(x)
        
        # Apply Conv2DBlock
        x = self.conv2d_block(x)

        # Apply Symmetrize2D
        x = self.symmetrize_2d(x)
        
        # Apply ResidualDilatedBlock - 2D
        x = self.residual2d_block1(x)
        x = self.residual2d_block2(x)
        x = self.residual2d_block3(x)
        x = self.residual2d_block4(x)
        x = self.residual2d_block5(x)
        x = self.residual2d_block6(x)
        
        # Apply Squeeze-and-Excite layer
        x = self.squeeze_excite(x)
        
        # Apply Cropping2D
        x = self.cropping_2d(x)
        
        # UpperTri
        # x = self.upper_tri(x)
        x = self.upper_tri(x, reverse_bool)
        
        # Add reversing matrices        
        x = self.final(x)
        
        return x

