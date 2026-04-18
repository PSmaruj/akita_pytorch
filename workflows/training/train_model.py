#!/usr/bin/env python3
"""
Train Akita v2 model from scratch.

This script trains an Akita model from random initialization on a Hi-C dataset.
It includes:
- Training from scratch (no pretrained weights)
- Early stopping based on validation loss
- Precise BatchNorm statistics updates
- Loss tracking and model checkpointing
"""

import argparse
import csv
import os
import sys

import schedulefree
import torch
from torch.utils.data import DataLoader

from akita.model import SeqNN
from data_preprocessing.dataset import HiCDataset
from training.training_utils import compute_initial_losses, train_epoch, validate

# =============================================================================
# Main Training Loop
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Akita v2 model from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory containing .pt files"
    )
    parser.add_argument(
        "--test_fold", type=str, required=True, help='Fold to use as test data (e.g., "fold0")'
    )
    parser.add_argument(
        "--val_fold", type=str, required=True, help='Fold to use as validation data (e.g., "fold1")'
    )

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer type"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.98, help="Momentum for SGD optimizer (ignored for Adam)"
    )
    parser.add_argument(
        "--l2-scale", type=float, default=1.5e-5, help="L2 regularization weight (weight_decay)"
    )
    parser.add_argument(
        "--weight-clipping", type=float, default=20.0, help="Weight clipping value (0 to disable)"
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Early stopping patience (epochs without improvement)",
    )

    # Logging and saving
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Batches between training status logs"
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="Save the best model checkpoint"
    )
    parser.add_argument(
        "--save-model-path", type=str, default="./model.pth", help="Path to save the best model"
    )
    parser.add_argument(
        "--save-losses", type=str, default="./losses.csv", help="Path to save loss history CSV"
    )

    # System arguments
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable CUDA training"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="Quick single-pass check"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    args = parser.parse_args()

    # ==========================================================================
    # Setup
    # ==========================================================================

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Akita v2 Training from Scratch")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Test fold: {args.test_fold}")
    print(f"Val fold: {args.val_fold}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Early stopping patience: {args.early_stop_patience}")
    print("=" * 70)
    print()

    # ==========================================================================
    # Load Data
    # ==========================================================================

    print("Loading data...")
    all_files = [
        os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".pt")
    ]

    test_files = [f for f in all_files if args.test_fold in f]
    val_files = [f for f in all_files if args.val_fold in f]
    train_files = [f for f in all_files if args.test_fold not in f and args.val_fold not in f]

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    train_dataset = HiCDataset(train_files)
    val_dataset = HiCDataset(val_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print()

    # ==========================================================================
    # Initialize Model
    # ==========================================================================

    print("Initializing model from scratch...")
    model = SeqNN()
    model.to(device)
    print("✓ Model initialized")
    print()

    # ==========================================================================
    # Setup Optimizer
    # ==========================================================================

    print(f"Setting up {args.optimizer} optimizer...")

    if args.optimizer == "adam":
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=args.lr, weight_decay=args.l2_scale
        )
    elif args.optimizer == "sgd":
        optimizer = schedulefree.SGDScheduleFree(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2_scale
        )

    print("✓ Optimizer configured")
    print()

    # ==========================================================================
    # Setup Output Paths
    # ==========================================================================

    # Create directories for output files
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_losses), exist_ok=True)

    print(f"Model checkpoint will be saved to: {args.save_model_path}")
    print(f"Loss history will be saved to: {args.save_losses}")
    print()

    # ==========================================================================
    # Compute Initial Losses
    # ==========================================================================

    print("Computing initial losses (before training)...")
    init_train_loss, init_val_loss = compute_initial_losses(model, device, train_loader, val_loader)

    print(f"Initial Train Loss: {init_train_loss:.6f}")
    print(f"Initial Validation Loss: {init_val_loss:.6f}")
    print()

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    print()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Open CSV file for logging
    file_exists = os.path.isfile(args.save_losses)
    with open(args.save_losses, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

        # Write initial losses (epoch 0)
        writer.writerow([0, init_train_loss, init_val_loss])
        f.flush()

        # Training epochs
        for epoch in range(1, args.epochs + 1):
            # Train
            train_loss = train_epoch(model, device, train_loader, optimizer, epoch, args)

            # Validate
            val_loss = validate(model, device, val_loader)

            # Log losses
            writer.writerow([epoch, train_loss, val_loss])
            f.flush()

            # Check for improvement
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                print(
                    f"✓ Validation loss improved: {best_val_loss:.6f} → {val_loss:.6f} "
                    f"(Δ {improvement:.6f})"
                )
                best_val_loss = val_loss
                epochs_no_improve = 0

                # Save best model
                if args.save_model:
                    torch.save(model.state_dict(), args.save_model_path)
                    print(f"✓ Model saved to {args.save_model_path}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")

            print()

            # Early stopping
            if epochs_no_improve >= args.early_stop_patience:
                print("=" * 70)
                print(f"Early stopping triggered after {epoch} epochs!")
                print(f"Best validation loss: {best_val_loss:.6f}")
                print("=" * 70)
                break

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Loss history saved to: {args.save_losses}")
    if args.save_model:
        print(f"Best model saved to: {args.save_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
