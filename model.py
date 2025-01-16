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
                     OneToTwo)


class SeqNN(nn.Module):
    def __init__(self, n_channel=4, max_len=128):
        super(SeqNN, self).__init__()

        # TRUNK
        
        self.stochastic_reverse_complement = StochasticReverseComplement()
        self.stochastic_shift = StochasticShift()
        self.re_lu = nn.ReLU()

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
            bn_momentum=0.9265,
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
            norm_type="batch",
            bn_momentum=0.9265
        )

        # ResidualDilatedBlock
        self.residual1d_block1 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=1
        )
        
        self.residual1d_block2 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=2
        )
        
        self.residual1d_block3 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=4
        )
        
        self.residual1d_block4 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=7
        )
        
        self.residual1d_block5 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=12
        )
        
        self.residual1d_block6 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=21
        )
        
        self.residual1d_block7 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=37
        )
        
        self.residual1d_block8 = ResidualDilatedBlock1D(
            in_channels=96,
            mid_channels=48,
            dropout_rate=0.4,
            dilation_rate=65
        )
        
        # ConvBlockReduce
        self.conv_reduce = ConvBlockReduce(
            in_channels=96,
            out_channels=64,
            kernel_size=5,
            bn_momentum=0.9265
        )
        
        # HEAD
        self.one_to_two = OneToTwo()
        
        
    def forward(self, x, training=False):
        device = x.device
        self.to(device)

        # Apply stochastic reverse complement
        x = self.stochastic_reverse_complement(x, training=training)

        # Apply stochastic shift
        x = self.stochastic_shift(x)

        # Apply ReLU activation
        x = self.re_lu(x)

        # Apply ConvBlock
        x = self.conv_block_1(x)

        # Apply ConvTower
        x = self.conv_tower(x)
        
        # Apply ResidualDilatedBlock
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
        
        print(f"End shape: {x.size()}")
        
        return x


# from torchsummary import summary

# class SeqNN(nn.Module):
#     def __init__(self,n_channel=4,max_len=128):
#         super(SeqNN, self).__init__()
        
#         self.stochastic_reverse_complement = StochasticReverseComplement()
#         self.stochastic_shift = StochasticShift()
#         self.re_lu = nn.ReLU()
        
#         # ConvBlock
#         self.conv_block_1 = ConvBlock(
#             filters=96,
#             kernel_size=11,
#             stride=1,
#             dilation_rate=1,
#             pool_size=2,
#             pool_type='max',
#             norm_type='batch',
#             bn_momentum=0.9265
#         )

#         # ConvTower
#         self.conv_tower = ConvTower(
#             in_channels=96,
#             filters_init=96,
#             filters_mult=1.0,
#             kernel_size=5,
#             pool_size=2,
#             repeat=10,
#             norm_type="batch",
#             bn_momentum=0.9265
#         )
        
    #     # DilatedResidual1D
    #     self.dilated_residual_block = DilatedResidual1D(
    #         in_channels=96,  # Same input channels as the output from ConvTower
    #         filters=48,      # Filters for dilated residual block
    #         kernel_size=3,   # Kernel size for convolutions
    #         rate_mult=1.75,  # Multiplier for dilation rate
    #         repeat=8,        # Number of dilated residual layers
    #         dropout=0.4,     # Dropout rate
    #         norm_type='batch',  # Batch normalization
    #     )
        
    # def forward(self,x, training=False):
        
    #     device = x.device
    #     self.to(device)
        
    #     x = self.stochastic_reverse_complement(x, training=training)
        
    #     x = self.stochastic_shift(x)
        
    #     x = self.re_lu(x)
        
    #     x = self.conv_block_1(x)
        
    #     x = self.conv_tower(x)
    #     print(f"After conv_tower: {x.size()}")

    #     # Apply DilatedResidual1D block
    #     x = self.dilated_residual_block(x)
    #     print(f"After dilated_residual_block: {x.size()}")
        
    #     return x



# def from_upptri(inputs):
#     seq_len = inputs.shape[2]
#     output_dim = inputs.shape[1]

#     triu_tup = np.triu_indices(seq_len, 2)
#     triu_index = list(triu_tup[0] + seq_len * triu_tup[1])
#     unroll_repr = inputs.reshape(-1, 1)
#     return torch.index_select(unroll_repr, 0,torch.tensor(triu_index))

# def calc_R_R2(y_true, y_pred, num_targets, device='cuda:0'):
#     '''
#     Handles the Pearson R and R2 calculation
#     '''
#     product = torch.sum(torch.multiply(y_true, y_pred), dim=1)
#     true_sum = torch.sum(y_true, dim=1)
#     true_sumsq = torch.sum(torch.square(y_true), dim=1)
#     pred_sum = torch.sum(y_pred, dim=1)
#     pred_sumsq = torch.sum(torch.square(y_pred), dim=1)
#     count = torch.sum(torch.ones(y_true.shape), dim=1).to(device)
#     true_mean = torch.divide(true_sum, count)
#     true_mean2 = torch.square(true_mean)

#     pred_mean = torch.divide(pred_sum, count)
#     pred_mean2 = torch.square(pred_mean)

    # term1 = product
    # term2 = -torch.multiply(true_mean, pred_sum)
    # term3 = -torch.multiply(pred_mean, true_sum)
    # term4 = torch.multiply(count, torch.multiply(true_mean, pred_mean))
    # covariance = term1 + term2 + term3 + term4

    # true_var = true_sumsq - torch.multiply(count, true_mean2)
    # pred_var = pred_sumsq - torch.multiply(count, pred_mean2)
    # pred_var = torch.where(torch.greater(pred_var, 1e-12), pred_var, np.inf*torch.ones(pred_var.shape).to(device))

    # tp_var = torch.multiply(torch.sqrt(true_var), torch.sqrt(pred_var))

    # correlation = torch.divide(covariance, tp_var)
    # correlation = correlation[~torch.isnan(correlation)]
    # correlation_mean = torch.mean(correlation)
    # total = torch.subtract(true_sumsq, torch.multiply(count, true_mean2))
    # resid1 = pred_sumsq
    # resid2 = -2*product
    # resid3 = true_sumsq
    # resid = resid1 + resid2 + resid3
    # r2 = torch.ones_like(torch.tensor(num_targets)) - torch.divide(resid, total)
    # r2 = r2[~torch.isinf(r2)]
    # r2_mean = torch.mean(r2)
    # return correlation_mean, r2_mean


