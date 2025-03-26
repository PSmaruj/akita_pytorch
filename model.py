import numpy as np
import random

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler,ConcatDataset
import torch.nn.functional as F

from sklearn.metrics import r2_score
from modules import (StochasticReverseComplement, 
                     StochasticShift, 
                     ConvBlock, 
                     ConvTower,
                     ResidualDilatedBlock1D, 
                     ConvBlockReduce, 
                     OneToTwo,
                     ConcatDist2D,
                     Conv2DBlock,
                     Symmetrize2D,
                     DilatedResidualBlock2D,
                     Cropping2D,
                     UpperTri,
                     Final)


class SeqNN(nn.Module):
    def __init__(self, n_channel=4):
        super(SeqNN, self).__init__()

        # TRUNK
        self.stochastic_reverse_complement = StochasticReverseComplement()
        self.stochastic_shift = StochasticShift(shift_max=11, symmetric=True, pad='constant')

        # ConvBlock
        self.conv_block_1 = ConvBlock(
            in_channels=n_channel,
            filters=96,
            kernel_size=11,
            stride=1,
            dilation_rate=1,
            pool_size=2,
            pool_type='max',
            norm_type='batch',
            bn_momentum=0.0735, # 1 - tensorflow momentum
            use_dropout=False
        )

        # ConvTower
        self.conv_tower = ConvTower(
            in_channels=96,
            filters_init=96,
            filters_mult=1.0,
            kernel_size=5,
            pool_size=2,
            repeat=10,
            norm_type='batch',
            bn_momentum=0.0735
        )

        # ResidualDilatedBlock
        self.residual1d_block1 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=1,
            norm_type="batch"
        )
        
        self.residual1d_block2 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=2,
            norm_type="batch"
        )
        
        self.residual1d_block3 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=4,
            norm_type="batch"
        )
        
        self.residual1d_block4 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=7,
            norm_type="batch"
        )
        
        self.residual1d_block5 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=12,
            norm_type="batch"
        )
        
        self.residual1d_block6 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=21,
            norm_type="batch"
        )
        
        self.residual1d_block7 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=37,
            norm_type="batch"
        )
        
        self.residual1d_block8 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=65,
            norm_type="batch"
        )
        
        # ConvBlockReduce
        self.conv_reduce = ConvBlockReduce(
            in_channels=96,
            out_channels=64,
            kernel_size=5,
            bn_momentum=0.0735,
            norm_type="batch"
        )
        
        # HEAD
        
        # 1D to 2D
        self.one_to_two = OneToTwo()
        
        # Concatenating distance
        self.concat_dist = ConcatDist2D()
        
        # ConvBlock
        self.conv2d_block = Conv2DBlock(
            in_channels=65, 
            out_channels=48, 
            kernel_size=3,
            norm_type="batch")
        
        # Symmetrize2D
        self.symmetrize_2d = Symmetrize2D()
        
        # ResidualDilatedBlock - 2D
        self.residual2d_block1 = DilatedResidualBlock2D(
            in_channels=48, 
            mid_channels=24, 
            kernel_size=3, 
            dilation_rate=1,
            dropout_prob=0.1,
            norm_type="batch")
        
        self.residual2d_block2 = DilatedResidualBlock2D(
            in_channels=48, 
            mid_channels=24, 
            kernel_size=3, 
            dilation_rate=2,
            dropout_prob=0.1,
            norm_type="batch")
        
        self.residual2d_block3 = DilatedResidualBlock2D(
            in_channels=48, 
            mid_channels=24, 
            kernel_size=3, 
            dilation_rate=4,
            dropout_prob=0.1,
            norm_type="batch")
        
        self.residual2d_block4 = DilatedResidualBlock2D(
            in_channels=48, 
            mid_channels=24, 
            kernel_size=3, 
            dilation_rate=7,
            dropout_prob=0.1,
            norm_type="batch")
        
        self.residual2d_block5 = DilatedResidualBlock2D(
            in_channels=48, 
            mid_channels=24, 
            kernel_size=3, 
            dilation_rate=12,
            dropout_prob=0.1,
            norm_type="batch")
        
        self.residual2d_block6 = DilatedResidualBlock2D(
            in_channels=48, 
            mid_channels=24, 
            kernel_size=3, 
            dilation_rate=21,
            dropout_prob=0.1,
            norm_type="batch")
        
        # Cropping2D
        # self.cropping_2d = Cropping2D(cropping=32)
        self.cropping_2d = Cropping2D(cropping=64)
        
        # UpperTri
        self.upper_tri = UpperTri()
        
        # Final
        self.final = Final(units=1, activation='linear')
        
    def forward(self, x):
        device = x.device
        self.to(device)

        # Apply stochastic reverse complement
        x, reverse_bool = self.stochastic_reverse_complement(x)

        # Apply stochastic shift
        x = self.stochastic_shift(x)

        # Apply ConvBlock
        x = self.conv_block_1(x)

        # Apply ConvTower
        x = self.conv_tower(x)
        
        # Apply ResidualDilatedBlock - 1D
        x = self.residual1d_block1(x) 
        x = self.residual1d_block2(x) 
        x = self.residual1d_block3(x)
        x = self.residual1d_block4(x)
        x = self.residual1d_block5(x)
        x = self.residual1d_block6(x)
        x = self.residual1d_block7(x)
        x = self.residual1d_block8(x)
        
        # Apply ConvBlockReduce
        x = self.conv_reduce(x)
        
        # Apply OneToTwo
        x = self.one_to_two(x)
        
        # Apply ConcatDist2D
        x = self.concat_dist(x)
        
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
        
        # Apply Cropping2D
        x = self.cropping_2d(x)
        
        # UpperTri
        x = self.upper_tri(x, reverse_bool)
        
        x = self.final(x)
        
        return x

