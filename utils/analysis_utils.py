"""
Analysis utilities for Akita v2.

This module contains functions for loading and analyzing training/validation
loss histories and model performance metrics.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_loss_file(organism, dataset_name, model_idx, base_path, training_type='finetuned'):
    """
    Load loss CSV file for a specific model.
    
    Args:
        organism (str): "mouse" or "human"
        dataset_name (str): Dataset name
        model_idx (int): Model split index
        base_path (str): Base path to models directory
        training_type (str): Type of training - 'finetuned' or 'trained_from_scratch'.
                           Default: 'finetuned'
    
    Returns:
        pd.DataFrame or None: Loss dataframe or None if file not found
    
    Example:
        >>> df = load_loss_file("mouse", "Hsieh2019_mESC", 0, "/path/to/models")
        >>> print(df.columns)
        Index(['Epoch', 'Train Loss', 'Validation Loss'], dtype='object')
    """
    file_path = (
        f"{base_path}/{training_type}/{organism}_models/{dataset_name}/losses/"
        f"Akita_v2_{organism}_{dataset_name}_model{model_idx}_{training_type}.csv"
    )

    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_all_models(organism, dataset_name, base_path, training_type='finetuned'):
    """
    Find all available model loss files for a dataset.
    
    Args:
        organism (str): "mouse" or "human"
        dataset_name (str): Dataset name
        base_path (str): Base path to models directory
        training_type (str): Type of training - 'finetuned' or 'trained_from_scratch'.
                           Default: 'finetuned'
    
    Returns:
        list: List of available model indices
    
    Example:
        >>> indices = find_all_models("mouse", "Hsieh2019_mESC", "/path/to/models")
        >>> print(indices)
        [0, 1, 2, 3, 4, 5, 6, 7]
    """
    pattern = (
        f"{base_path}/{training_type}/{organism}_models/{dataset_name}/losses/"
        f"Akita_v2_{organism}_{dataset_name}_model*_{training_type}.csv"
    )

    files = glob.glob(pattern)

    # Extract model indices from filenames
    model_indices = []
    for f in files:
        try:
            # Extract number between "model" and f"_{training_type}"
            basename = os.path.basename(f)
            idx_str = basename.split("model")[1].split(f"_{training_type}")[0]
            model_indices.append(int(idx_str))
        except:
            continue

    return sorted(model_indices)


def plot_single_model(df, model_idx, dataset_name, figsize=(10, 5)):
    """
    Plot loss curves for a single model.
    
    Args:
        df (pd.DataFrame): Loss dataframe with columns ['Epoch', 'Train Loss', 'Validation Loss']
        model_idx (int): Model index
        dataset_name (str): Dataset name for title
        figsize (tuple): Figure size. Default: (10, 5)
    
    Returns:
        tuple: (fig, ax, best_epoch, best_val_loss)
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df['Epoch'], df['Train Loss'],
            label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax.plot(df['Epoch'], df['Validation Loss'],
            label='Validation Loss', linewidth=2, marker='s', markersize=3)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(f'Loss Curves: {dataset_name} (Model {model_idx})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Find best epoch (minimum validation loss)
    best_epoch = df.loc[df['Validation Loss'].idxmin(), 'Epoch']
    best_val_loss = df['Validation Loss'].min()
    ax.axvline(x=best_epoch, color='red', linestyle=':',
               label=f'Best (Epoch {best_epoch:.0f})', alpha=0.7)
    ax.legend(fontsize=11)

    plt.tight_layout()

    return fig, ax, best_epoch, best_val_loss


def plot_all_models(organism, dataset_name, model_indices, base_path,
                    training_type='finetuned', figsize_per_plot=6):
    """
    Plot loss curves for all models in a grid of subplots.
    
    Args:
        organism (str): "mouse" or "human"
        dataset_name (str): Dataset name
        model_indices (list): List of model indices to plot
        base_path (str): Base path to models directory
        training_type (str): Type of training. Default: 'finetuned'
        figsize_per_plot (float): Size of each subplot. Default: 6
    
    Returns:
        tuple: (fig, axes, summary_stats) where summary_stats is a dict with
               'best_losses', 'best_epochs', 'best_model_idx'
    """
    n_models = len(model_indices)

    # Determine grid layout
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot * n_cols, 4 * n_rows)
    )

    # Handle single subplot case
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Track statistics
    all_best_epochs = []
    all_best_losses = []

    for i, model_idx in enumerate(model_indices):
        ax = axes[i]

        # Load data
        df = load_loss_file(organism, dataset_name, model_idx, base_path, training_type)

        if df is None:
            ax.text(0.5, 0.5, f'Model {model_idx}\nData not found',
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot
        ax.plot(df['Epoch'], df['Train Loss'],
               label='Train', linewidth=2, alpha=0.8)
        ax.plot(df['Epoch'], df['Validation Loss'],
               label='Validation', linewidth=2, alpha=0.8)

        # Mark best epoch
        best_epoch = df.loc[df['Validation Loss'].idxmin(), 'Epoch']
        best_val_loss = df['Validation Loss'].min()
        ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.5)

        all_best_epochs.append(best_epoch)
        all_best_losses.append(best_val_loss)

        # Formatting
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss (MSE)', fontsize=10)
        ax.set_title(
            f'Model {model_idx}\n(Best: {best_val_loss:.5f} @ Epoch {best_epoch:.0f})',
            fontsize=11, fontweight='bold'
        )
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Loss Curves: {dataset_name}',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Summary statistics
    summary_stats = {
        'best_losses': all_best_losses,
        'best_epochs': all_best_epochs,
        'best_model_idx': model_indices[np.argmin(all_best_losses)] if all_best_losses else None,
        'worst_model_idx': model_indices[np.argmax(all_best_losses)] if all_best_losses else None,
        'mean_loss': np.mean(all_best_losses) if all_best_losses else None,
        'std_loss': np.std(all_best_losses) if all_best_losses else None,
        'mean_epoch': np.mean(all_best_epochs) if all_best_epochs else None,
        'std_epoch': np.std(all_best_epochs) if all_best_epochs else None
    }

    return fig, axes, summary_stats


def print_summary_statistics(dataset_name, model_indices, summary_stats):
    """
    Print summary statistics for model training results.
    
    Args:
        dataset_name (str): Dataset name
        model_indices (list): List of model indices
        summary_stats (dict): Dictionary with summary statistics from plot_all_models
    """
    print("\n" + "="*70)
    print(f"Summary Statistics for {dataset_name}")
    print("="*70)
    print(f"Number of models: {len(model_indices)}")

    if summary_stats['best_model_idx'] is not None:
        print(f"Best validation loss: {min(summary_stats['best_losses']):.6f} "
              f"(Model {summary_stats['best_model_idx']})")
        print(f"Worst validation loss: {max(summary_stats['best_losses']):.6f} "
              f"(Model {summary_stats['worst_model_idx']})")
        print(f"Mean validation loss: {summary_stats['mean_loss']:.6f} ± "
              f"{summary_stats['std_loss']:.6f}")
        print(f"Mean best epoch: {summary_stats['mean_epoch']:.1f} ± "
              f"{summary_stats['std_epoch']:.1f}")
    else:
        print("No valid data found")

    print("="*70)


def compare_datasets(organism, dataset_names, base_path, training_type='finetuned'):
    """
    Compare best validation losses across multiple datasets.
    
    Args:
        organism (str): "mouse" or "human"
        dataset_names (list): List of dataset names to compare
        base_path (str): Base path to models directory
        training_type (str): Type of training. Default: 'finetuned'
    
    Returns:
        pd.DataFrame: Comparison dataframe with columns
                     ['Dataset', 'Model', 'Best Val Loss']
    """
    comparison_data = []

    for dataset in dataset_names:
        model_indices = find_all_models(organism, dataset, base_path, training_type)

        for model_idx in model_indices:
            df = load_loss_file(organism, dataset, model_idx, base_path, training_type)
            if df is not None:
                best_val_loss = df['Validation Loss'].min()
                comparison_data.append({
                    'Dataset': dataset,
                    'Model': model_idx,
                    'Best Val Loss': best_val_loss
                })

    return pd.DataFrame(comparison_data)


def plot_dataset_comparison(comparison_df, figsize=(12, 6)):
    """
    Plot comparison of best validation losses across datasets.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame from compare_datasets
        figsize (tuple): Figure size. Default: (12, 6)
    
    Returns:
        tuple: (fig, ax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    datasets = comparison_df['Dataset'].unique()

    for dataset in datasets:
        data = comparison_df[comparison_df['Dataset'] == dataset]
        ax.plot(data['Model'], data['Best Val Loss'],
                marker='o', linewidth=2, label=dataset, markersize=8)

    ax.set_xlabel('Model Index', fontsize=12)
    ax.set_ylabel('Best Validation Loss', fontsize=12)
    ax.set_title('Comparison of Best Validation Losses Across Datasets',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    return fig, ax
