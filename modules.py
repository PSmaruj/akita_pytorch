
import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one-hot encoded DNA sequence."""
    
    def __init__(self):
        super(StochasticReverseComplement, self).__init__()

    def forward(self, seq_1hot):
        device = seq_1hot.device  # Ensure tensors are on the same device
        
        if self.training:
            # Reverse complement: rearrange channels (A->T, C->G, G->C, T->A)
            rc_seq_1hot = seq_1hot.index_select(dim=1, index=torch.tensor([3, 2, 1, 0], device=device))
            
            # Flip the sequence along the sequence axis (dim=-1)
            rc_seq_1hot = torch.flip(rc_seq_1hot, dims=[-1])
            
            # Create a random boolean tensor for batch decision (batch_size,)
            reverse_bool = torch.rand(seq_1hot.size(0), device=device) > 0.5
            
            # Use reverse_bool to select between original and reversed sequence
            result = torch.where(reverse_bool[:, None, None], rc_seq_1hot, seq_1hot)
        else:
            # In eval mode, keep the sequence unchanged
            result = seq_1hot
            reverse_bool = torch.zeros(seq_1hot.size(0), device=device, dtype=torch.bool)
        
        # Return both the modified sequence and the reverse decision flag
        return result, reverse_bool


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
        # Create a tensor of all possible shift values
        if self.symmetric:
            self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)  # Including 0
        else:
            self.augment_shifts = torch.arange(0, self.shift_max + 1)  # Only positive shifts
        self.pad = pad

    def shift_sequence(self, seq_1hot, shift):
        """
        Applies a shift to a sequence by padding and slicing accordingly.
        """
        if shift > 0:
            seq_1hot_padded = F.pad(seq_1hot, (0, shift, 0, 0), mode=self.pad)  # Pad at the beginning
            shifted_seq_1hot = seq_1hot_padded[:, shift:]  # Remove the extra part at the end
        elif shift < 0:
            seq_1hot_padded = F.pad(seq_1hot, (-shift, 0, 0, 0), mode=self.pad)  # Pad at the end
            shifted_seq_1hot = seq_1hot_padded[:, :shift]  # Remove the extra part at the beginning
        else:
            shifted_seq_1hot = seq_1hot  # No shift if shift == 0
        return shifted_seq_1hot

    def forward(self, seq_1hot):
        """
        Applies a random shift to the input sequence during training, or returns the sequence unchanged during inference.
        """
        if self.training:
            device = seq_1hot.device  # Ensure tensors match device
            self.augment_shifts = self.augment_shifts.to(device)  # Move augment_shifts to the same device
            
            # Pick a random shift for each sequence in the batch
            shift_indices = torch.randint(len(self.augment_shifts), size=(seq_1hot.size(0),), device=device)
            shifts = self.augment_shifts[shift_indices]  # A batch of random shift values
            
            # Apply the shift to each sequence in the batch
            shifted_seq_1hot = torch.stack([
                self.shift_sequence(seq_1hot[i], shifts[i]) for i in range(seq_1hot.size(0))
            ])
            return shifted_seq_1hot
        else:
            return seq_1hot


class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1, dilation_rate=1, pool_size=1, pool_type='max', 
                 norm_type=None, bn_momentum=0.0735, dropout_prob=0.4, use_dropout=True):
        super(ConvBlock, self).__init__()

        # Convolution Layer
        self.conv = nn.Conv1d(in_channels, filters, kernel_size, stride=stride, padding=(kernel_size // 2), dilation=dilation_rate, bias=False)

        # Both in pytorch and tensorflow, gammas (weights) are initialized as 1s and betas (biases) as 0s, by default.
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(filters, eps=0.001, momentum=bn_momentum) if norm_type == 'batch' else None

        # group normalization
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=filters, eps=0.001) if norm_type == 'group' else None
        
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
        
        if self.group_norm:
            x = self.group_norm(x)
        
        # Apply pooling if specified
        if self.pool:
            x = self.pool(x)
        
        # Apply dropout if specified
        if self.use_dropout:
            x = self.dropout(x)
        
        return x


class ConvTower(nn.Module):
    def __init__(self, in_channels, filters_init, filters_mult, kernel_size, pool_size, repeat, norm_type="batch", bn_momentum=0.0735):
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
                layers.append(nn.BatchNorm1d(int(filters), eps=0.001, momentum=bn_momentum))
            elif norm_type == "group":
                layers.append(nn.GroupNorm(num_groups=32, num_channels=int(filters), eps=0.001))
                
            # Pooling
            layers.append(nn.MaxPool1d(kernel_size=pool_size))

            # Update filters for the next layer
            filters *= filters_mult

        self.conv_tower = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_tower(x)


class ResidualDilatedBlock1D(nn.Module):
    def __init__(self, in_channels, mid_channels, dropout_rate=0.4, dilation_rate=1, bn_momentum=0.0735, norm_type='batch'):
        super(ResidualDilatedBlock1D, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels, mid_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate, bias=False
        )
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(mid_channels, eps=0.001, momentum=bn_momentum)
        elif norm_type == 'group':
            num_groups = mid_channels // 2
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=mid_channels, eps=0.001)
        else:
            self.norm1 = None
        
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(
            mid_channels, in_channels, kernel_size=1, padding=0, bias=False
        )
        if norm_type == 'batch':
            self.norm2 = nn.BatchNorm1d(in_channels, eps=0.001, momentum=bn_momentum)
        elif norm_type == 'group':
            num_groups = in_channels // 2
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=0.001)
        else:
            self.norm2 = None  # No normalization if norm_type is None or unsupported value
        
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
    def __init__(self, in_channels, out_channels, kernel_size=5, bn_momentum=0.0735, norm_type='batch', num_groups=32):
        super(ConvBlockReduce, self).__init__()
        
        layers = [
            nn.ReLU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,  # To preserve sequence length
                bias=False
            )
        ]
        
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(out_channels, eps=0.001, momentum=bn_momentum))
        elif norm_type == 'group':
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=0.001))
        
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

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
        # dist = torch.abs(matrix_repr1 - matrix_repr2).to(torch.float64) # to set a particular precision
        dist = dist.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, seq_len, seq_len]

        # Concatenate along the feature axis
        return torch.cat([inputs, dist], dim=1)


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_momentum=0.0735, norm_type='batch'):
        super(Conv2DBlock, self).__init__()
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        ]
        
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(out_channels, eps=0.001, momentum=bn_momentum))
        elif norm_type == 'group':
            num_groups = out_channels // 2
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=0.001))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    

class Symmetrize2D(nn.Module):
    """Take the average of a matrix and its transpose to enforce symmetry."""
    def __init__(self):
        super(Symmetrize2D, self).__init__()

    def forward(self, x):
        # Transpose the last two dimensions (height and width for 2D data)
        x_t = torch.transpose(x, 2, 3)  # Swap dimensions 2 and 3
        # Compute the symmetric average
        x_sym = (x + x_t) / 2
        return x_sym   


class DilatedResidualBlock2D(nn.Module):
    def __init__(self, in_channels=48, mid_channels=24, kernel_size=3, dilation_rate=1, dropout_prob=0.1, bn_momentum=0.0735,
                 norm_type='batch'):
        """
        A dilated residual block with symmetry enforcement.
        Args:
            in_channels (int): Number of input and output channels.
            mid_channels (int): Number of intermediate channels.
            kernel_size (int): Kernel size for the convolutional layers.
            dilation_rate (int): Dilation rate for the convolutional layers.
            dropout_prob (float): Dropout probability.
        """
        super(DilatedResidualBlock2D, self).__init__()
        self.relu = nn.ReLU()
        
        # First convolutional layer (reduces channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2) * dilation_rate,
            dilation=dilation_rate,  # Add dilation
            bias=False
        )
        if norm_type == 'batch':
            self.norm1 = nn.BatchNorm2d(mid_channels, eps=0.001, momentum=bn_momentum)
        elif norm_type == 'group':
            num_groups = mid_channels // 2
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=mid_channels, eps=0.001)
        else:
            self.norm1 = None  # No normalization if norm_type is None or unsupported value

        # Second convolutional layer (restores original channel count)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=in_channels,
            kernel_size=1,
            padding=(1 // 2) * dilation_rate,  # Adjust padding for dilation
            dilation=dilation_rate,  # Add dilation
            bias=False
        )
        if norm_type == 'batch':
            self.norm2 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=bn_momentum)
        elif norm_type == 'group':
            num_groups = in_channels // 2
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=0.001)
        else:
            self.norm2 = None
        
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Symmetrization layer
        self.symmetrize = Symmetrize2D()

    def forward(self, x):
        # Save residual input
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
        
        # Dropout
        x = self.dropout(x)

        # Add residual connection
        x = x + residual

        # Symmetrize
        x = self.symmetrize(x)

        return x


class Cropping2D(nn.Module):
    def __init__(self, cropping):
        super(Cropping2D, self).__init__()
        self.cropping = cropping

    def forward(self, inputs):
        _, _, h, w = inputs.size()  # Get the height and width of the input tensor
        cropped = inputs[:, :, self.cropping:h - self.cropping, self.cropping:w - self.cropping]
        return cropped


class UpperTri(nn.Module):
    ''' Unroll matrix to its upper triangular portion. '''
    def __init__(self, diagonal_offset=2):
        super(UpperTri, self).__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, inputs, reverse_complement_flags):
        # Get the batch size, features_dim, and mat_size from the shape of the input tensor
        batch_size, features_dim, mat_size, _ = inputs.shape
        
        # Generate the upper triangular indices with the specified diagonal offset
        triu_tup = torch.triu_indices(mat_size, mat_size, self.diagonal_offset)

        # Flatten the input tensor to shape [batch_size, features_dim, mat_size^2]
        unroll_repr = inputs.reshape(batch_size, features_dim, mat_size * mat_size)
        
        # Convert triu indices to flattened indices
        triu_index = triu_tup[0] * mat_size + triu_tup[1]  # Row * mat_size + col
        triu_index = triu_index.to(inputs.device)
        
        # Unsqueeze triu_index to make it [mat_size^2] and expand it across all the batch examples and channels
        triu_index = triu_index.unsqueeze(0).unsqueeze(0).expand(batch_size, features_dim, -1)

        # Generate the flipped version of the input by rotating along the main diagonal
        flipped_repr = unroll_repr.flip(2)  # Flip along the flattened matrix rows

        unroll_repr = unroll_repr.to(inputs.device)
        flipped_repr = flipped_repr.to(inputs.device)  
        reverse_complement_flags = reverse_complement_flags.to(inputs.device)
        
        # Use reverse_complement_flags to select between original and flipped maps
        upper_tri = torch.where(reverse_complement_flags.view(batch_size, 1, 1).expand(batch_size, features_dim, -1),
                                torch.gather(unroll_repr, 2, triu_index),
                                torch.gather(flipped_repr, 2, triu_index))

        return upper_tri


class Final(nn.Module):
    def __init__(self, l2_scale=0, l1_scale=0, activation='linear', units=5, **kwargs):
        super(Final, self).__init__()
        self.l2_scale = l2_scale
        self.l1_scale = l1_scale
        self.activation = activation
        self.units = units
        
        # Dense layer to map seq_len (48) to new_seq_len (5)
        self.dense = nn.Linear(in_features=48, out_features=self.units) #, bias=True)  # Transform channels (seq_len) only

    def forward(self, x):
        # x dim: batch_size, feature_dim, seq_length
        
        # Apply the dense layer along the feature_dim (which will transform it to num_units)
        x = self.dense(x.transpose(1, 2))

        # Transpose back to [batch_size, num_units, seq_length]
        x = x.transpose(1, 2)

        # Apply activation function
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation == 'linear':
            pass  # No activation (linear is default)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        # Optionally apply regularization (L1 and L2)
        if self.l2_scale > 0:
            x = F.normalize(x, p=2, dim=-1)  # L2 normalization

        if self.l1_scale > 0:
            x = F.normalize(x, p=1, dim=-1)  # L1 normalization

        return x


# class SwitchReverseTriu(nn.Module):
#     def __init__(self, diagonal_offset, matrix_size):
#         """
#         Args:
#             diagonal_offset (int): Offset for the diagonal in the upper triangular matrix.
#             matrix_size (int): Size of the square matrix (e.g., 448 for 448x448).
#         """
#         super(SwitchReverseTriu, self).__init__()
    #     self.diagonal_offset = diagonal_offset
    #     self.matrix_size = matrix_size  # Square matrix size
    #     self.ut_len = (matrix_size * (matrix_size + 1)) // 2  # Total upper triangular elements

    # def forward(self, x, reverse_bool):
    #     """
    #     Forward pass with optional reversal of the upper triangular indices.

    #     Args:
    #         x (Tensor): Input tensor with shape [batch_size, channels, ut_len].
    #         reverse_bool (Tensor): Boolean tensor of shape [batch_size] indicating reversal.

    #     Returns:
    #         Tensor: Processed tensor with the same shape as input.
    #     """
    #     batch_size, channels, length = x.size()

    #     # Get upper triangular indices for the matrix
    #     ut_indices = torch.triu_indices(self.matrix_size, self.matrix_size, self.diagonal_offset).to(x.device)

    #     # Flip the elements in the upper triangle based on reverse_bool
    #     if reverse_bool.any():
    #         # Reverse the triangular elements (swap upper triangle elements)
    #         for b in range(batch_size):
    #             if reverse_bool[b]:
    #                 # Get the elements for the upper triangle in the current batch
    #                 upper_triangle = x[b, :, :]  # [channels, ut_len]
    #                 flipped = torch.flip(upper_triangle, dims=[1])  # Flip along the last dimension (upper triangle)
    #                 x[b, :, :] = flipped

    #     return x
