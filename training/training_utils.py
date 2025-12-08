"""
Shared training utilities for Akita v2 model.

This module contains common functions used by both training from scratch
and fine-tuning scripts.
"""

import torch
import torch.nn.functional as F
from fvcore.nn.precise_bn import update_bn_stats

# =============================================================================
# Utility Functions
# =============================================================================

def data_loader_for_precise_bn(loader, device):
    """
    Generator wrapper for precise BatchNorm statistics computation.

    Args:
        loader (DataLoader): Training data loader
        device: PyTorch device

    Yields:
        torch.Tensor: Input data batches
    """
    for data, _ in loader:
        yield data.to(device)


def compute_loss(output, target):
    """
    Compute MSE loss, ignoring NaN values in target.

    Args:
        output (torch.Tensor): Model predictions
        target (torch.Tensor): Ground truth (may contain NaNs)

    Returns:
        torch.Tensor: MSE loss computed only on valid (non-NaN) entries
    """
    valid_mask = ~torch.isnan(target)
    if not valid_mask.any():
        # All targets are NaN - return zero loss
        return torch.tensor(0.0, device=output.device)
    return F.mse_loss(output[valid_mask], target[valid_mask])


# =============================================================================
# Training and Validation Functions
# =============================================================================

def train_epoch(model, device, train_loader, optimizer, epoch, args):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        device: PyTorch device
        train_loader: Training data loader
        optimizer: Optimizer (schedulefree.AdamWScheduleFree or SGDScheduleFree)
        epoch (int): Current epoch number
        args: Command line arguments (must have log_interval, dry_run,
              weight_clipping attributes)

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    optimizer.train()

    total_loss = 0.0
    num_batches = 0
    total_nans = 0
    total_vals = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward pass and compute loss
        output = model(data)
        loss = compute_loss(output, target)

        # Track NaN statistics
        total_nans += torch.isnan(target).sum().item()
        total_vals += target.numel()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Apply weight clipping
        if args.weight_clipping > 0:
            for param in model.parameters():
                param.data.clamp_(-args.weight_clipping, args.weight_clipping)

        total_loss += loss.item()
        num_batches += 1

        # Logging
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            if args.dry_run:
                break

    # Report NaN statistics
    nan_frac = total_nans / total_vals if total_vals > 0 else 0
    print(f'Training NaN fraction: {nan_frac:.2%} ({total_nans}/{total_vals})')

    avg_loss = total_loss / max(num_batches, 1)

    # Update BatchNorm statistics with precise estimation
    print('Updating BatchNorm statistics with preciseBN...')
    update_bn_stats(
        model,
        data_loader_for_precise_bn(train_loader, device),
        num_iters=min(len(train_loader), 200)
    )

    return avg_loss


def validate(model, device, val_loader):
    """
    Validate model on validation set.

    Args:
        model: PyTorch model
        device: PyTorch device
        val_loader: Validation data loader

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_nans = 0
    total_vals = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Track NaN statistics
            total_nans += torch.isnan(target).sum().item()
            total_vals += target.numel()

            loss = compute_loss(output, target)
            if loss.item() > 0:  # Only count batches with valid data
                total_loss += loss.item()
                num_batches += 1

    # Report NaN statistics
    nan_frac = total_nans / total_vals if total_vals > 0 else 0
    print(f'Validation NaN fraction: {nan_frac:.2%} ({total_nans}/{total_vals})')

    avg_loss = total_loss / max(num_batches, 1)
    print(f'Validation set: Average MSE loss: {avg_loss:.4f}\n')

    return avg_loss


def compute_initial_losses(model, device, train_loader, val_loader):
    """
    Compute initial training and validation losses before training starts.

    Args:
        model: PyTorch model
        device: PyTorch device
        train_loader: Training data loader
        val_loader: Validation data loader

    Returns:
        tuple: (train_loss, val_loss)
    """
    model.eval()

    with torch.no_grad():
        # Initial training loss
        init_train_loss = 0.0
        num_train_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = compute_loss(output, target)
            if loss.item() > 0:
                init_train_loss += loss.item()
                num_train_batches += 1
        init_train_loss /= max(num_train_batches, 1)

        # Initial validation loss
        init_val_loss = 0.0
        num_val_batches = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = compute_loss(output, target)
            if loss.item() > 0:
                init_val_loss += loss.item()
                num_val_batches += 1
        init_val_loss /= max(num_val_batches, 1)

    return init_train_loss, init_val_loss
