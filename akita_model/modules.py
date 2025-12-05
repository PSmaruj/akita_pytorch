"""
Neural network modules for the Akita v2 architecture.

This module contains all the building blocks used in the Akita v2 model for
genome folding prediction, including:
- Data augmentation layers (StochasticReverseComplement, StochasticShift)
- 1D convolutional blocks (ConvBlock, ConvTower, ResidualDilatedBlock1D)
- 1D to 2D transformation layers (OneToTwo, ConcatDist2D)
- 2D convolutional blocks (Conv2DBlock, DilatedResidualBlock2D)
- Attention mechanisms (SqueezeExcite)
- Output processing layers (UpperTri, Final)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticReverseComplement(nn.Module):
    """
    Stochastically reverse complement a one-hot encoded DNA sequence.
    
    During training, randomly applies reverse complement transformation to DNA
    sequences (A<->T, C<->G, flipped left-to-right). During evaluation, sequences
    are returned unchanged.
    
    Args:
        None
    
    Input shape:
        (batch_size, 4, sequence_length) - one-hot encoded DNA
    
    Output shape:
        tuple: (transformed_sequence, reverse_bool)
            - transformed_sequence: (batch_size, 4, sequence_length)
            - reverse_bool: (batch_size,) boolean tensor indicating which sequences
              were reverse complemented
    """
    
    def __init__(self):
        super(StochasticReverseComplement, self).__init__()

    def forward(self, seq_1hot):
        device = seq_1hot.device  # Ensure tensors are on the same device
        
        if self.training:
            # Reverse complement: rearrange channels (A->T, C->G, G->C, T->A)
            # Channels are [A, C, G, T] -> reverse to [T, G, C, A]
            rc_seq_1hot = seq_1hot.index_select(
                dim=1, 
                index=torch.tensor([3, 2, 1, 0], device=device)
            )
            
            # Flip the sequence along the sequence axis
            rc_seq_1hot = torch.flip(rc_seq_1hot, dims=[-1])
            
            # Randomly decide which sequences to reverse complement
            reverse_bool = torch.rand(seq_1hot.size(0), device=device) > 0.5
            
            # Apply reverse complement to selected sequences
            result = torch.where(
                reverse_bool[:, None, None], 
                rc_seq_1hot, 
                seq_1hot
            )
        else:
            # In eval mode, keep the sequence unchanged
            result = seq_1hot
            reverse_bool = torch.zeros(
                seq_1hot.size(0), 
                device=device, 
                dtype=torch.bool
            )
        
        # Return both the modified sequence and the reverse decision flag
        return result, reverse_bool


class StochasticShift(nn.Module):
    """
    Applies random horizontal shifts to DNA sequences during training.
    
    Sequences are padded and shifted within a specified range. This data
    augmentation helps the model learn translation-invariant features.
    
    Args:
        shift_max (int): Maximum shift distance in base pairs. Default: 0
        symmetric (bool): If True, allows shifts in both directions [-shift_max, shift_max].
                         If False, only positive shifts [0, shift_max]. Default: True
        pad (str): Padding mode for out-of-bounds regions. Options: 'constant',
                  'reflect', etc. Default: 'constant'
    
    Input shape:
        (batch_size, 4, sequence_length)
    
    Output shape:
        (batch_size, 4, sequence_length)
    """
    def __init__(self, shift_max=0, symmetric=True, pad='constant'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        self.pad = pad
        
        # Create a tensor of all possible shift values
        if self.symmetric:
            self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
        else:
            self.augment_shifts = torch.arange(0, self.shift_max + 1)

    def shift_sequence(self, seq_1hot, shift):
        """
        Apply a shift to a single sequence.
        
        Args:
            seq_1hot (Tensor): Sequence to shift
            shift (int): Shift amount (positive = right, negative = left)
        
        Returns:
            Tensor: Shifted sequence
        """
        if shift > 0:
            # Shift right: pad left, remove right
            seq_1hot_padded = F.pad(seq_1hot, (0, shift, 0, 0), mode=self.pad)
            shifted_seq_1hot = seq_1hot_padded[:, shift:]
        elif shift < 0:
            # Shift left: pad right, remove left
            seq_1hot_padded = F.pad(seq_1hot, (-shift, 0, 0, 0), mode=self.pad)
            shifted_seq_1hot = seq_1hot_padded[:, :shift]
        else:
            # No shift if shift == 0
            shifted_seq_1hot = seq_1hot
        return shifted_seq_1hot

    def forward(self, seq_1hot):
        """
        Applies a random shift to the input sequence during training, or returns the sequence unchanged during inference.
        """
        if not self.training:
            return seq_1hot
        
        device = seq_1hot.device
        self.augment_shifts = self.augment_shifts.to(device)
                    
        # Sample random shifts for each sequence in batch
        shift_indices = torch.randint(
            len(self.augment_shifts), 
            size=(seq_1hot.size(0),), 
            device=device
        )
        shifts = self.augment_shifts[shift_indices]
        
        # Apply shifts
        shifted_seq_1hot = torch.stack([
            self.shift_sequence(seq_1hot[i], shifts[i]) 
            for i in range(seq_1hot.size(0))
        ])
        
        return shifted_seq_1hot


class ConvBlock(nn.Module):
    """
    Basic 1D convolutional block with normalization, pooling, and dropout.
    
    Args:
        in_channels (int): Number of input channels
        filters (int): Number of output filters
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride. Default: 1
        dilation_rate (int): Dilation rate for dilated convolutions. Default: 1
        pool_size (int): Pooling kernel size. Default: 1 (no pooling)
        pool_type (str): Type of pooling ('max' or None). Default: 'max'
        norm_type (str): Normalization type ('batch' or None). Default: None
        bn_momentum (float): Momentum for batch normalization. Default: 0.1
        dropout_prob (float): Dropout probability. Default: 0.4
        use_dropout (bool): Whether to apply dropout. Default: True
    """
    
    def __init__(self, in_channels, filters, kernel_size, stride=1, 
                 dilation_rate=1, pool_size=1, pool_type='max', 
                 norm_type=None, bn_momentum=0.1, dropout_prob=0.4, 
                 use_dropout=True):
        super(ConvBlock, self).__init__()

        # Convolution Layer
        self.conv = nn.Conv1d(
            in_channels, 
            filters, 
            kernel_size, 
            stride=stride, 
            padding=(kernel_size // 2), 
            dilation=dilation_rate, 
            bias=False)

        # Normalization
        self.batch_norm = nn.BatchNorm1d(
            filters, 
            eps=0.001, 
            momentum=bn_momentum
        ) if norm_type == 'batch' else None
        
        # Pooling
        self.pool = nn.MaxPool1d(pool_size) if pool_type == 'max' else None

        # Dropout (Optional)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=dropout_prob) if self.use_dropout else None

    def forward(self, x):
        x = self.conv(x)
        
        if self.batch_norm:
            x = self.batch_norm(x)
    
        if self.pool:
            x = self.pool(x)
        
        if self.use_dropout:
            x = self.dropout(x)
        
        return x


class ConvTower(nn.Module):
    """
    Stack of convolutional layers with increasing filter counts.
    
    Each layer consists of: ReLU -> Conv1D -> Normalization -> MaxPool.
    Filter counts increase by filters_mult at each layer.
    
    Args:
        in_channels (int): Number of input channels
        filters_init (int): Initial number of filters
        filters_mult (float): Multiplier for filters at each layer
        kernel_size (int): Convolution kernel size
        pool_size (int): Pooling kernel size
        repeat (int): Number of layers in the tower
        norm_type (str): Normalization type ('batch' or None). Default: 'batch'
        bn_momentum (float): Batch normalization momentum. Default: 0.1
    """
    
    def __init__(self, in_channels, filters_init, filters_mult, kernel_size, 
                 pool_size, repeat, norm_type='batch', bn_momentum=0.1):
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
                padding=kernel_size // 2,
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
    """
    1D residual block with dilated convolutions.
    
    Architecture: x -> ReLU -> Conv(dilation) -> Norm -> ReLU -> Conv(1x1) -> 
                  Norm -> Dropout -> Add residual
    
    Args:
        in_channels (int): Number of input/output channels
        mid_channels (int): Number of intermediate channels (bottleneck)
        dropout_rate (float): Dropout probability. Default: 0.4
        dilation_rate (int): Dilation rate for first convolution. Default: 1
        bn_momentum (float): Batch normalization momentum. Default: 0.1
        norm_type (str): Normalization type ('batch' or None). Default: 'batch'
    """
    
    def __init__(self, in_channels, mid_channels, dropout_rate=0.4, 
                 dilation_rate=1, bn_momentum=0.1, norm_type='batch'):
        super(ResidualDilatedBlock1D, self).__init__()
        
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels, 
            mid_channels, 
            kernel_size=3, 
            padding=dilation_rate, 
            dilation=dilation_rate, 
            bias=False
        )
        
        self.norm1 = nn.BatchNorm1d(
            mid_channels, 
            eps=0.001, 
            momentum=bn_momentum
        ) if norm_type == 'batch' else None
        
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            mid_channels, 
            in_channels, 
            kernel_size=1, 
            padding=0, 
            bias=False
        )
        
        self.norm2 = nn.BatchNorm1d(
            in_channels, 
            eps=0.001, 
            momentum=bn_momentum
        ) if norm_type == 'batch' else None
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        
        out = self.relu1(x)
        out = self.conv1(out)
        if self.norm1:
            out = self.norm1(out)
        
        out = self.relu2(out)
        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)
        
        out = self.dropout(out)
        out += residual
        
        return out


class ConvBlockReduce(nn.Module):
    """
    1D convolutional block for channel reduction.
    
    Architecture: ReLU -> Conv1D -> Normalization -> ReLU
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size. Default: 5
        bn_momentum (float): Batch normalization momentum. Default: 0.1
        norm_type (str): Normalization type ('batch', 'group', or None). Default: 'batch'
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=5, 
                 bn_momentum=0.1, norm_type='batch'):
        super(ConvBlockReduce, self).__init__()
        
        layers = [
            nn.ReLU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False
            )
        ]
        
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(
                out_channels, 
                eps=0.001, 
                momentum=bn_momentum
            ))
        
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OneToTwo(nn.Module):
    """
    Transform 1D sequence features into 2D pairwise features.
    
    Creates a 2D feature map by combining each position with every other position.
    Different operations control how position pairs are combined.
    
    Args:
        operation (str): How to combine position pairs. Options:
            - 'concat': Concatenate features from both positions
            - 'mean': Average features from both positions
            - 'max': Maximum features from both positions
            - 'multiply': Element-wise multiplication
            - 'multiply1': (x+1)*(y+1)-1 transformation
            Default: 'mean'
    
    Input shape:
        (batch_size, features, seq_len)
    
    Output shape:
        (batch_size, output_features, seq_len, seq_len)
        where output_features = 2*features for 'concat', else features
    """
    
    def __init__(self, operation='mean'):
        super(OneToTwo, self).__init__()
        self.operation = operation.lower()
        valid_operations = ['concat', 'mean', 'max', 'multiply', 'multiply1']
        assert self.operation in valid_operations, \
            f"Invalid operation '{operation}'. Choose from {valid_operations}"

    def forward(self, oned):
        batch_size, features, seq_len = oned.shape

        # Create 2D representations by expanding and reshaping
        twod1 = oned.repeat(1, 1, seq_len).view(batch_size, features, seq_len, seq_len)
        twod2 = twod1.permute(0, 1, 3, 2)  # Transpose spatial dimensions

        if self.operation == 'concat':
            twod = torch.cat([twod1, twod2], dim=1)
        elif self.operation == 'multiply':
            twod = twod1 * twod2
        elif self.operation == 'multiply1':
            twod = (twod1 + 1) * (twod2 + 1) - 1
        else:
            # For mean/max operations
            twod1 = twod1.unsqueeze(-1)
            twod2 = twod2.unsqueeze(-1)
            twod = torch.cat([twod1, twod2], dim=-1)

            if self.operation == 'mean':
                twod = twod.mean(dim=-1)
            elif self.operation == 'max':
                twod, _ = twod.max(dim=-1)

        return twod


class ConcatDist2D(nn.Module):
    """
    Concatenate pairwise genomic distance to 2D feature matrix.
    
    Adds a channel containing the distance between each pair of positions,
    which helps the model learn distance-dependent patterns in Hi-C data.
    
    Input shape:
        (batch_size, features, seq_len, seq_len)
    
    Output shape:
        (batch_size, features+1, seq_len, seq_len)
    """

    def __init__(self):
        super(ConcatDist2D, self).__init__()

    def forward(self, inputs):
        batch_size, features, seq_len, seq_len_ = inputs.shape

        assert seq_len == seq_len_, \
            f"Input must be square, got ({seq_len}, {seq_len_})"

        # Create pairwise distance matrix
        pos = torch.arange(seq_len, device=inputs.device)
        pos = pos.unsqueeze(0).repeat(seq_len, 1)
        dist = torch.abs(pos - pos.t()).float()
        
        # Expand to batch
        dist = dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Concatenate distance channel
        return torch.cat([inputs, dist], dim=1)


class Conv2DBlock(nn.Module):
    """
    Basic 2D convolutional block with normalization.
    
    Architecture: ReLU -> Conv2D -> Normalization
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size. Default: 3
        bn_momentum (float): Batch normalization momentum. Default: 0.1
        norm_type (str): Normalization type ('batch' or None). Default: 'batch'
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 bn_momentum=0.1, norm_type='batch'):
        super(Conv2DBlock, self).__init__()
        
        layers = [
            nn.ReLU(),
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                bias=False
                )
        ]
        
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(
                out_channels, 
                eps=0.001, 
                momentum=bn_momentum
            ))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    

class Symmetrize2D(nn.Module):
    """
    Enforce matrix symmetry by averaging with transpose.
    
    This is important for Hi-C contact matrices which represent symmetric
    pairwise interactions between genomic positions.
    
    Input/Output shape:
        (batch_size, channels, height, width)
    """
    
    def __init__(self):
        super(Symmetrize2D, self).__init__()

    def forward(self, x):
        x_t = torch.transpose(x, 2, 3)  # Transpose spatial dimensions
        x_sym = (x + x_t) / 2
        return x_sym   


class DilatedResidualBlock2D(nn.Module):
    """
    2D residual block with dilated convolutions and symmetry enforcement.
    
    Architecture: x -> ReLU -> Conv2D(dilation) -> Norm -> ReLU -> Conv2D(1x1) -> 
                  Norm -> Dropout -> Add residual -> Symmetrize
    
    Args:
        in_channels (int): Number of input/output channels. Default: 48
        mid_channels (int): Number of intermediate channels (bottleneck). Default: 24
        kernel_size (int): Kernel size for first convolution. Default: 3
        dilation_rate (int): Dilation rate for first convolution. Default: 1
        dropout_prob (float): Dropout probability. Default: 0.1
        bn_momentum (float): Batch normalization momentum. Default: 0.1
        norm_type (str): Normalization type ('batch' or None). Default: 'batch'
    """
    
    def __init__(self, in_channels=48, mid_channels=24, kernel_size=3, 
                 dilation_rate=1, dropout_prob=0.1, bn_momentum=0.1,
                 norm_type='batch'):
        super(DilatedResidualBlock2D, self).__init__()
        
        self.relu = nn.ReLU()
        
        # First convolutional layer (channel reduction with dilation)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2) * dilation_rate,
            dilation=dilation_rate,  # Add dilation
            bias=False
        )
        
        self.norm1 = nn.BatchNorm2d(
            mid_channels, 
            eps=0.001, 
            momentum=bn_momentum
        ) if norm_type == 'batch' else None

        # Second convolutional layer (channel restoration)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=in_channels,
            kernel_size=1,
            padding=(1 // 2) * dilation_rate,
            dilation=dilation_rate,
            bias=False
        )
        
        self.norm2 = nn.BatchNorm2d(
            in_channels, 
            eps=0.001, 
            momentum=bn_momentum
        ) if norm_type == 'batch' else None
        
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.symmetrize = Symmetrize2D()

    def forward(self, x):
        residual = x

        # First convolutional block
        x = self.relu(x)
        x = self.conv1(x)
        if self.norm1:
            x = self.norm1(x)

        # Second convolutional block
        x = self.relu(x)
        x = self.conv2(x)
        if self.norm2:
            x = self.norm2(x)
        
        # Dropout and residual connection
        x = self.dropout(x)
        x = x + residual

        # Enforce symmetry
        x = self.symmetrize(x)

        return x


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Adaptively recalibrates channel-wise features by explicitly modeling
    interdependencies between channels.
    
    Args:
        in_channels (int): Number of input channels
        activation (str): Activation function ('relu', 'gelu', or 'silu'). Default: 'relu'
        additive (bool): If True, add attention weights; if False, multiply. Default: False
        bottleneck_ratio (int): Reduction ratio for bottleneck. Default: 8
        norm_type (str): Normalization type ('batch' or None). Default: None
        bn_momentum (float): Batch normalization momentum. Default: 0.9
    
    Input shape:
        (batch_size, channels, height, width)
    
    Output shape:
        (batch_size, channels, height, width)
    """
    
    def __init__(self, in_channels, activation='relu', additive=False, 
                 bottleneck_ratio=8, norm_type=None, bn_momentum=0.9):
        super(SqueezeExcite, self).__init__()
        self.activation = activation
        self.additive = additive
        self.norm_type = norm_type
        self.bn_momentum = bn_momentum
        self.in_channels = in_channels

        # Squeeze: Global Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Excitation: MLP with bottleneck
        reduced_channels = max(in_channels // bottleneck_ratio, 1)
        self.dense1 = nn.Linear(in_channels, reduced_channels)
        self.dense2 = nn.Linear(reduced_channels, in_channels)

        # Normalization
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(in_channels, eps=0.001, momentum=bn_momentum)
        else:
            self.norm = None

    def forward(self, x):
        device = x.device

        # Apply activation
        x = self._activate(x)

        # Squeeze: Global Average Pooling
        batch_size, channels, _, _ = x.shape
        squeeze = self.global_pool(x).view(batch_size, channels)
        squeeze = squeeze.to(device)

        # Excite: MLP
        excite = self.dense1(squeeze)
        excite = F.relu(excite)
        excite = self.dense2(excite)

        if self.norm is not None:
            excite = self.norm(excite)

        # Reshape and apply attention
        excite = excite.view(batch_size, channels, 1, 1)
        
        if self.additive:
            x = x + excite
        else:
            excite = torch.sigmoid(excite)
            x = x * excite

        return x

    def _activate(self, x):
        """Apply specified activation function."""
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        elif self.activation == 'silu':
            return F.silu(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")



class Cropping2D(nn.Module):
    """
    Crop 2D feature maps symmetrically from all sides.
    
    Args:
        cropping (int): Number of pixels to crop from each side
    
    Input shape:
        (batch_size, channels, height, width)
    
    Output shape:
        (batch_size, channels, height-2*cropping, width-2*cropping)
    """
    
    def __init__(self, cropping):
        super(Cropping2D, self).__init__()
        self.cropping = cropping

    def forward(self, inputs):
        _, _, h, w = inputs.size()
        cropped = inputs[
            :, 
            :, 
            self.cropping:h - self.cropping, 
            self.cropping:w - self.cropping
        ]
        return cropped


class UpperTri(nn.Module):
    """
    Extract upper triangular portion of contact matrix and handle reverse complement.
    
    For Hi-C matrices, we only need the upper triangle since they're symmetric.
    Also handles reversing the triangle when the input sequence was reverse complemented.
    
    Args:
        diagonal_offset (int): Offset from main diagonal (2 = skip first 2 diagonals).
                              Default: 2
    
    Input shape:
        - inputs: (batch_size, features, mat_size, mat_size)
        - reverse_complement_flags: (batch_size,) boolean tensor
    
    Output shape:
        (batch_size, features, num_upper_tri_elements)
    """
    
    def __init__(self, diagonal_offset=2):
        super(UpperTri, self).__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, inputs, reverse_complement_flags):
        batch_size, features_dim, mat_size, _ = inputs.shape

        # Compute both flipped and unflipped versions
        flipped_inputs = inputs.transpose(-1, -2).flip(-1).flip(-2)

        # Select version based on reverse_complement_flags
        transformed_inputs = torch.where(
            reverse_complement_flags.view(-1, 1, 1, 1),  # Expand dims for broadcasting
            flipped_inputs,
            inputs
        )
        
        # Generate the upper triangular indices
        triu_tup = torch.triu_indices(
            mat_size, 
            mat_size, 
            self.diagonal_offset, 
            device=inputs.device
        )
    
        # Convert to flattened indices
        triu_index = triu_tup[0] * mat_size + triu_tup[1]
        triu_index = triu_index.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 
            features_dim, 
            -1
        )

        # Flatten input tensor and extract upper triangle
        unroll_repr = transformed_inputs.reshape(
            batch_size, 
            features_dim, 
            mat_size * mat_size
        )
        upper_tri = torch.gather(unroll_repr, 2, triu_index)

        return upper_tri


class Final(nn.Module):
    """
    Final output layer that transforms features to target predictions.
    
    Maps from feature dimension to the desired number of output units (targets).
    
    Args:
        activation (str): Output activation function. Options: 'relu', 'gelu', 'linear'.
                         Default: 'linear'
        units (int): Number of output units/targets. Default: 5
    
    Input shape:
        (batch_size, 80, seq_length)
    
    Output shape:
        (batch_size, units, seq_length)
    
    Note:
        For L2 regularization, use weight_decay parameter in your optimizer.
    """
    
    def __init__(self, activation='linear', units=5, **kwargs):
        super(Final, self).__init__()
        self.activation = activation
        self.units = units
        
        # Dense layer maps from input features (80) to output units
        self.dense = nn.Linear(in_features=80, out_features=self.units)

    def forward(self, x):
        # x shape: (batch_size, feature_dim, seq_length)
        
        # Apply dense transformation along feature dimension
        x = self.dense(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # Apply activation function
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation == 'linear':
            pass  # No activation
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        return x


class SwitchReverseTriu(nn.Module):
    """
    Reverse upper triangular elements based on reverse complement flags.
    
    This module is used when the model needs to handle reverse complemented
    sequences after extracting the upper triangular portion.
    
    Args:
        diagonal_offset (int): Diagonal offset used in upper triangle extraction
        matrix_size (int): Size of the original square matrix (e.g., 448)
    
    Input shape:
        - x: (batch_size, channels, ut_len) where ut_len is upper triangle length
        - reverse_bool: (batch_size,) boolean tensor
    
    Output shape:
        (batch_size, channels, ut_len)
    """
    
    def __init__(self, diagonal_offset, matrix_size):
        super(SwitchReverseTriu, self).__init__()
        self.diagonal_offset = diagonal_offset
        self.matrix_size = matrix_size
        self.ut_len = (matrix_size * (matrix_size + 1)) // 2

    def forward(self, x, reverse_bool):
        batch_size, channels, length = x.size()

        # Get upper triangular indices
        ut_indices = torch.triu_indices(
            self.matrix_size, 
            self.matrix_size, 
            self.diagonal_offset
        ).to(x.device)

        # Flip elements based on reverse_bool
        if reverse_bool.any():
            for b in range(batch_size):
                if reverse_bool[b]:
                    upper_triangle = x[b, :, :]
                    flipped = torch.flip(upper_triangle, dims=[1])
                    x[b, :, :] = flipped

        return x
