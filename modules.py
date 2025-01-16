import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
import torch.nn.functional as F

from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast


# from torchsummary import summary
from sklearn.metrics import r2_score, precision_score, f1_score

# from ray import tune

import json
import itertools
from itertools import groupby
import gzip
from io import BytesIO
from time import time

import matplotlib.pyplot as plt

import pyBigWig
from scipy.sparse import csc_matrix
import math
from copy import deepcopy


class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one hot encoded DNA sequence."""
    
    def __init__(self):
        super(StochasticReverseComplement, self).__init__()

    def forward(self, seq_1hot, training=True):
        device = seq_1hot.device  # Ensure tensors are on the same device
        
        # Print the input shape of seq_1hot
        print(f"Input seq_1hot shape: {seq_1hot.shape}")
        
        if training:
            # Reverse complement: rearrange channels (A->T, C->G, G->C, T->A)
            # The rearrangement is done across the 1st dimension (channel dimension)
            rc_seq_1hot = seq_1hot.index_select(dim=1, index=torch.tensor([3, 2, 1, 0], device=device))
            
            # Flip the sequence along the sequence axis (dim=-1), keeping the channel order intact
            rc_seq_1hot = torch.flip(rc_seq_1hot, dims=[-1])  # Flip along the sequence axis (dim=-1)
            
            # Create a random boolean tensor to decide whether to reverse, shape: (batch_size,)
            reverse_bool = torch.rand(seq_1hot.size(0), device=device) > 0.5
            
            # Expand reverse_bool to match the shape of seq_1hot (batch_size, 1, 1)
            reverse_bool = reverse_bool.unsqueeze(1).unsqueeze(2)
            
            # Expand reverse_bool across the channels (dim 1) and sequence length (dim 2)
            reverse_bool = reverse_bool.expand(-1, seq_1hot.size(1), seq_1hot.size(2))
            
            # Conditionally return the reversed complement sequence based on reverse_bool
            result = torch.where(reverse_bool, rc_seq_1hot, seq_1hot)
        else:
            # If not training, no need to stochastically reverse, return seq_1hot unchanged
            result = seq_1hot
        
        return result


class StochasticShift(nn.Module):
    """
    Applies a random shift to a one-hot encoded DNA sequence during training.

    The class allows for a random horizontal shift of the sequence within a given range, where the sequence 
    is padded to handle the shift. The shift can be symmetric (allowing shifts in both directions) or 
    asymmetric (only shifts in the positive direction). The padding mode can be specified (default is 'constant').

    Attributes:
        shift_max (int): Maximum shift range (both positive and negative if symmetric).
        symmetric (bool): If True, allows shifts in both directions (symmetric shift); otherwise, only positive shifts are applied.
        pad (str): Padding mode for out-of-bounds areas, e.g., 'constant' or 'reflect'.
        augment_shifts (tensor): Tensor of possible shift values based on the symmetric/asymmetric setting.

    Methods:
        shift_sequence(seq_1hot, shift): Shifts the input sequence by the specified amount, padding the sequence as needed.
        forward(seq_1hot): Randomly applies a shift to the input sequence during training, or returns the sequence unchanged during inference.
    """
    def __init__(self, shift_max=0, symmetric=True, pad='constant'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
        else:
            self.augment_shifts = torch.arange(0, self.shift_max + 1)
        self.pad = pad

    def shift_sequence(self, seq_1hot, shift):
        if shift > 0:
            seq_1hot_padded = F.pad(seq_1hot, (0, 0, shift, 0), mode=self.pad)
            shifted_seq_1hot = seq_1hot_padded[:, :-shift, :]
        else:
            seq_1hot_padded = F.pad(seq_1hot, (0, 0, 0, -shift), mode=self.pad)
            shifted_seq_1hot = seq_1hot_padded[:, -shift:, :]
        return shifted_seq_1hot

    def forward(self, seq_1hot):
        if self.training:
            device = seq_1hot.device  # Ensure tensors match device
            self.augment_shifts = self.augment_shifts.to(device)  # Move augment_shifts to the same device
            shift_i = torch.randint(len(self.augment_shifts), size=(1,), device=device)
            shift = self.augment_shifts[shift_i]
            sseq_1hot = torch.where(shift != 0,
                                    self.shift_sequence(seq_1hot, shift),
                                    seq_1hot)
            return sseq_1hot
        else:
            return seq_1hot


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1, dilation_rate=1, pool_size=1, pool_type='max', 
                 norm_type=None, bn_momentum=0.99, dropout_prob=0.4, use_dropout=True):
        super(ConvBlock, self).__init__()
        
        # Convolution Layer
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, stride=stride, padding=(kernel_size // 2), dilation=dilation_rate, bias=False)
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(filters, momentum=bn_momentum) if norm_type == 'batch' else None
        
        # Pooling (MaxPool)
        self.pool = nn.MaxPool1d(pool_size) if pool_type == 'max' else None
        
        # Dropout (Optional)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=dropout_prob) if self.use_dropout else None

    def forward(self, x):
        x = self.conv(x)
        
        # Apply Batch Normalization if specified
        if self.batch_norm:
            x = self.batch_norm(x)
        
        x = F.relu(x)
        
        # Apply pooling if specified
        if self.pool:
            x = self.pool(x)
        
        # Apply dropout if specified
        if self.use_dropout:
            x = self.dropout(x)
            
        return x


class ConvTower(nn.Module):
    def __init__(self, in_channels, filters_init, filters_mult, kernel_size, pool_size, repeat, norm_type="batch", bn_momentum=0.9265):
        super(ConvTower, self).__init__()
        
        layers = []
        filters = filters_init

        for i in range(repeat):
            # Activation
            layers.append(nn.ReLU())

            # Convolution
            layers.append(nn.Conv1d(
                in_channels=in_channels if i == 0 else int(filters),
                out_channels=int(filters),
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Same padding
                bias=False
            ))

            # Normalization
            if norm_type == "batch":
                layers.append(nn.BatchNorm1d(int(filters), momentum=bn_momentum))

            # Pooling
            layers.append(nn.MaxPool1d(kernel_size=pool_size))

            # Update filters for the next layer
            filters *= filters_mult

        self.conv_tower = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_tower(x)


class ResidualDilatedBlock1D(nn.Module):
    def __init__(self, in_channels, mid_channels, dropout_rate=0.1, dilation_rate=1, bn_momentum=0.9265):
        super(ResidualDilatedBlock1D, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels, mid_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, bias=False
        )
        self.bn1 = nn.BatchNorm1d(mid_channels, momentum=bn_momentum)
        
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            mid_channels, in_channels, kernel_size=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm1d(in_channels, momentum=bn_momentum)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.dropout(out)
        out += residual
        
        return out


class ConvBlockReduce(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, bn_momentum=0.9):
        super(ConvBlockReduce, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,  # To preserve sequence length
                bias=False
            ),
            nn.BatchNorm1d(out_channels, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class OneToTwo(nn.Module):
    def __init__(self, operation='mean'):
        super(OneToTwo, self).__init__()
        self.operation = operation.lower()
        valid_operations = ['concat', 'mean', 'max', 'multiply', 'multiply1']
        assert self.operation in valid_operations, f"Invalid operation. Choose from {valid_operations}"

    def forward(self, oned):
        batch_size, features, seq_len = oned.shape

        # Expand and reshape to create twod1 and twod2
        twod1 = oned.repeat(1, 1, seq_len).view(batch_size, features, seq_len, seq_len)
        twod2 = twod1.permute(0, 1, 3, 2)  # Swap dimensions 2 and 3

        if self.operation == 'concat':
            twod = torch.cat([twod1, twod2], dim=1)  # Concatenate along the feature dimension
        elif self.operation == 'multiply':
            twod = twod1 * twod2
        elif self.operation == 'multiply1':
            twod = (twod1 + 1) * (twod2 + 1) - 1
        else:
            # Expand last dimension for mean/max operations
            twod1 = twod1.unsqueeze(-1)
            twod2 = twod2.unsqueeze(-1)
            twod = torch.cat([twod1, twod2], dim=-1)

            if self.operation == 'mean':
                twod = twod.mean(dim=-1)  # Reduce mean along the last dimension
            elif self.operation == 'max':
                twod, _ = twod.max(dim=-1)  # Reduce max along the last dimension

        return twod


class ConcatDist2D(nn.Module):
    '''Concatenate the pairwise distance to 2D feature matrix.'''

    def __init__(self):
        super(ConcatDist2D, self).__init__()

    def forward(self, inputs):
        # Assume inputs is of shape [batch_size, features, seq_len, seq_len]
        batch_size, features, seq_len, seq_len_ = inputs.shape

        # Check input consistency
        assert seq_len == seq_len_, "Input dimensions must be square (seq_len, seq_len)."

        # Create pairwise distance matrix
        pos = torch.arange(seq_len, device=inputs.device).unsqueeze(0).repeat(seq_len, 1)  # [seq_len, seq_len]
        matrix_repr1 = pos
        matrix_repr2 = pos.t()
        dist = torch.abs(matrix_repr1 - matrix_repr2).float()  # [seq_len, seq_len]
        dist = dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, seq_len, seq_len]

        # Concatenate along the feature axis
        return torch.cat([inputs, dist], dim=1)


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_momentum=0.9265):
        super(Conv2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        )
    
    def forward(self, x):
        return self.block(x)
    
    

################################


# class Symmetrize2D(nn.Module):
#     def __init__(self):
#         super(Symmetrize2D, self).__init__()

#     def forward(self, x):
#         x_t = torch.transpose(x, 2,3)
#         x_sym = (x + x_t) / 2
#         return x_sym

# class DilatedResidual2D(nn.Module):
#     def __init__(self, in_channels, kernel_size, rate_mult, repeat, dropout):
#         super(DilatedResidual2D, self).__init__()
#         self.dropout = nn.Dropout2d(p=dropout)

#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)

#         # Define dilations
#         dilations = [1]
#         for i in range(1, repeat):
#             dilations.append(int(i*rate_mult))

#         # Define residual blocks
#         self.res_blocks = nn.ModuleList()
#         for dilation in dilations:
#             self.res_blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size, dilation=dilation, padding=dilation))

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = F.relu(out)
    #     out = self.dropout(out)

    #     out = self.conv2(out)
    #     out = F.relu(out)
    #     out = self.dropout(out)

    #     # Residual block
    #     for block in self.res_blocks:
    #         res = out
    #         out = F.relu(out)
    #         out = self.dropout(out)

    #         out = block(out)
    #         out = F.relu(out)
    #         out = self.dropout(out)

    #         out = out + res

    #     return out

# class Cropping2D(nn.Module):
#     def __init__(self, cropping):
#         super(Cropping2D, self).__init__()
#         self.cropping = cropping

#     def forward(self, inputs):
#         _, _, h, w = inputs.size()
#         cropped = inputs[:, :, self.cropping:h-self.cropping, self.cropping:w-self.cropping]
#         return cropped


# class UpperTri(nn.Module):
#     def __init__(self, diagonal_offset=2):
#         super(UpperTri, self).__init__()
#         self.diagonal_offset = diagonal_offset

#     def forward(self, inputs):
#         seq_len = inputs.shape[2]
#         output_dim = inputs.shape[1]

#         triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
#         triu_index = list(triu_tup[0]+ seq_len*triu_tup[1])
#         unroll_repr = inputs.reshape(-1, output_dim, seq_len**2)
#         return torch.index_select(unroll_repr, 2, torch.tensor(triu_index))

#     def extra_repr(self):
#         return 'diagonal_offset={}'.format(self.diagonal_offset)


# class Final(nn.Module):
#     def __init__(self, l2_scale=0, l1_scale=0, **kwargs):
#         super(Final, self).__init__()
#         # self.flatten = nn.Flatten()
#         self.l2_scale = l2_scale
#         self.l1_scale = l1_scale
#         self.dense = nn.Linear(in_features=12,out_features=1,bias=False)
#     def forward(self,x):
#         # x = self.flatten(x)
#         # print(x.size())
#         x = self.dense(x)
#         # regularize
#         if self.l2_scale > 0:
#             x = F.normalize(x, p=2, dim=-1)
#         if self.l1_scale > 0:
#             x = F.normalize(x, p=1, dim=-1)

#         return x

# def final(inputs, units, activation='linear', flatten=False,
#           kernel_initializer='he_normal', l2_scale=0, l1_scale=0, **kwargs):
#     """Final simple transformation before comparison to targets.
#     Args:
#         inputs:         [batch_size, seq_length, features] input sequence
#         units:          Dense units
#         activation:     relu/gelu/etc
#         flatten:        Flatten positional axis.
#         l2_scale:       L2 regularization weight.
#         l1_scale:       L1 regularization weight.
#     Returns:
#         [batch_size, seq_length(?), units] output sequence
#     """
#     current = inputs
#
#     # flatten
#     if flatten:
#         batch_size, seq_len, seq_depth = current.size()
#         current = current.view(batch_size, 1, seq_len * seq_depth)
#
#     # dense
#     current = nn.Linear(
#         in_features=current.size(-1),
#         out_features=units,
#         bias=True
#     )(current)
#     if activation == 'relu':
#         current = F.relu(current)
#     elif activation == 'gelu':
#         current = F.gelu(current)
#     elif activation == 'sigmoid':
#         current = torch.sigmoid(current)
#     elif activation == 'tanh':
#         current = torch.tanh(current)
#
#     # regularize
#     if l2_scale > 0:
#         current = F.normalize(current,p=2, dim=-1)
#     if l1_scale > 0:
#         current = F.normalize(current,p=1, dim=-1)
#
#     return current


# class PearsonR(nn.Module):
#     def __init__(self, num_targets, summarize=True):
#         super(PearsonR, self).__init__()
#         self.summarize = summarize
#         self.shape = (num_targets,)
#         self.count = nn.Parameter(torch.zeros(self.shape))

#         self.product = nn.Parameter(torch.zeros(self.shape))
#         self.true_sum = nn.Parameter(torch.zeros(self.shape))
#         self.true_sumsq = nn.Parameter(torch.zeros(self.shape))
#         self.pred_sum = nn.Parameter(torch.zeros(self.shape))
#         self.pred_sumsq = nn.Parameter(torch.zeros(self.shape))

    # def forward(self, y_true, y_pred):
    #     y_true = y_true.float()
    #     y_pred = y_pred.float()

    #     if len(y_true.shape) == 2:
    #         reduce_axes = 0
    #     else:
    #         reduce_axes = [0,1]

    #     product = torch.sum(torch.mul(y_true, y_pred), dim=reduce_axes)
    #     self.product.data.add_(product)

    #     true_sum = torch.sum(y_true, dim=reduce_axes)
    #     self.true_sum.data.add_(true_sum)

    #     true_sumsq = torch.sum(torch.pow(y_true, 2), dim=reduce_axes)
    #     self.true_sumsq.data.add_(true_sumsq)

    #     pred_sum = torch.sum(y_pred, dim=reduce_axes)
    #     self.pred_sum.data.add_(pred_sum)

    #     pred_sumsq = torch.sum(torch.pow(y_pred, 2), dim=reduce_axes)
    #     self.pred_sumsq.data.add_(pred_sumsq)

    #     count = torch.ones_like(y_true)
    #     count = torch.sum(count, dim=reduce_axes)
    #     self.count.data.add_(count)

    # def result(self):
    #     true_mean = torch.div(self.true_sum, self.count)
    #     true_mean2 = torch.pow(true_mean, 2)
    #     pred_mean = torch.div(self.pred_sum, self.count)
    #     pred_mean2 = torch.pow(pred_mean, 2)

    #     term1 = self.product
    #     term2 = -torch.mul(true_mean, self.pred_sum)
    #     term3 = -torch.mul(pred_mean, self.true_sum)
    #     term4 = torch.mul(self.count, torch.mul(true_mean, pred_mean))
    #     covariance = term1 + term2 + term3 + term4

    #     true_var = self.true_sumsq - torch.mul(self.count, true_mean2)
    #     pred_var = self.pred_sumsq - torch.mul(self.count, pred_mean2)
    #     pred_var = torch.where(pred_var > 1e-12, pred_var, torch.full_like(pred_var, float('inf')))

    #     tp_var = torch.mul(torch.sqrt(true_var), torch.sqrt(pred_var))
    #     correlation = torch.div(covariance, tp_var)

    #     if self.summarize:
    #         return torch.mean(correlation)
    #     else:
    #         return correlation

    # def reset_state(self):
    #     self.product.data.fill_(0)
    #     self.true_sum.data.fill_(0)
    #     self.true_sumsq.data.fill_(0)
    #     self.pred_sum.data.fill_(0)
    #     self.pred_sumsq.data.fill_(0)
    #     self.count.data.fill_(0)

