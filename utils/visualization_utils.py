"""
Visualization utilities for Akita v2.

This module contains shared visualization functions used across notebooks
for plotting Hi-C contact matrices and other visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_contact_map(
    matrix, vmin=-0.6, vmax=0.6, palette="RdBu_r", width=5, height=5, title=None, show_colorbar=True
):
    """
    Plot a Hi-C contact matrix as a heatmap.

    Args:
        matrix (np.ndarray): Contact matrix to plot
        vmin (float): Minimum value for color scale. Default: -0.6
        vmax (float): Maximum value for color scale. Default: 0.6
        palette (str): Seaborn color palette name. Default: "RdBu_r"
        width (float): Figure width in inches. Default: 5
        height (float): Figure height in inches. Default: 5
        title (str): Optional title for the plot. Default: None
        show_colorbar (bool): Whether to show colorbar. Default: True

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(1, 1, figsize=(width, height))

    cbar_kws = {"label": "Log(Obs/Exp)"} if show_colorbar else None

    sns.heatmap(
        matrix,
        vmin=vmin,
        vmax=vmax,
        cbar=show_colorbar,
        cmap=palette,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws=cbar_kws,
    )

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()

    return fig, ax


def set_diag(matrix, value, k):
    """
    Set a diagonal of a matrix to a specific value.

    Args:
        matrix (np.ndarray): Input matrix (modified in-place)
        value: Value to set diagonal elements to
        k (int): Diagonal offset
            - k=0: main diagonal
            - k>0: diagonal above main diagonal
            - k<0: diagonal below main diagonal

    Example:
        >>> matrix = np.ones((3, 3))
        >>> set_diag(matrix, 0, 0)  # Set main diagonal to 0
        >>> matrix
        array([[0., 1., 1.],
               [1., 0., 1.],
               [1., 1., 0.]])
    """
    rows, cols = matrix.shape
    for i in range(rows):
        if 0 <= i + k < cols:
            matrix[i, i + k] = value


def upper_triu_to_matrix(vector_repr, matrix_len=512, num_diags=2):
    """
    Convert upper triangular vector representation back to full symmetric matrix.

    Args:
        vector_repr (torch.Tensor or np.ndarray): Upper triangular vector
        matrix_len (int): Size of output square matrix. Default: 512
        num_diags (int): Number of diagonals that were skipped. Default: 2

    Returns:
        np.ndarray: Symmetric contact matrix with near-diagonal set to NaN

    Note:
        The first num_diags diagonals are set to NaN as they often contain
        artifacts from Hi-C data processing.

    Example:
        >>> vector = torch.randn(130305)  # Upper triangle of 512x512 matrix
        >>> matrix = upper_triu_to_matrix(vector, matrix_len=512, num_diags=2)
        >>> matrix.shape
        (512, 512)
    """
    # Convert to numpy if needed
    if isinstance(vector_repr, torch.Tensor):
        vector_repr = vector_repr.detach().flatten().cpu().numpy()

    # Initialize zero matrix
    matrix = np.zeros((matrix_len, matrix_len))

    # Get upper triangular indices (skipping diagonals)
    triu_indices = np.triu_indices(matrix_len, num_diags)

    # Fill upper triangle
    matrix[triu_indices] = vector_repr

    # Set skipped diagonals to NaN
    for i in range(-num_diags + 1, num_diags):
        set_diag(matrix, np.nan, i)

    # Make symmetric
    symmetric_matrix = matrix + matrix.T

    return symmetric_matrix


def plot_comparison(target_matrix, pred_matrix, sample_idx=0, vmin=-0.6, vmax=0.6):
    """
    Plot ground truth and prediction side by side.

    Args:
        target_matrix (np.ndarray): Ground truth contact matrix
        pred_matrix (np.ndarray): Predicted contact matrix
        sample_idx (int): Sample index for title. Default: 0
        vmin (float): Minimum value for color scale. Default: -0.6
        vmax (float): Maximum value for color scale. Default: 0.6

    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ground truth
    sns.heatmap(
        target_matrix,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cmap="RdBu_r",
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=axes[0],
        cbar_kws={"label": "Log(Obs/Exp)"},
    )
    axes[0].set_title(f"Ground Truth (Sample {sample_idx})", fontsize=12, fontweight="bold")

    # Prediction
    sns.heatmap(
        pred_matrix,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cmap="RdBu_r",
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=axes[1],
        cbar_kws={"label": "Log(Obs/Exp)"},
    )
    axes[1].set_title(f"Prediction (Sample {sample_idx})", fontsize=12, fontweight="bold")

    plt.tight_layout()

    return fig, axes


def plot_matrix_grid(
    matrices, titles=None, n_cols=3, vmin=-0.6, vmax=0.6, suptitle=None, figsize_per_plot=4
):
    """
    Plot multiple contact matrices in a grid.

    Args:
        matrices (list): List of contact matrices to plot
        titles (list): Optional list of titles for each subplot. Default: None
        n_cols (int): Number of columns in grid. Default: 3
        vmin (float): Minimum value for color scale. Default: -0.6
        vmax (float): Maximum value for color scale. Default: 0.6
        suptitle (str): Overall title for the figure. Default: None
        figsize_per_plot (float): Size of each subplot. Default: 4

    Returns:
        tuple: (fig, axes) matplotlib figure and axes objects
    """
    n_samples = len(matrices)
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize_per_plot * n_cols, figsize_per_plot * n_rows)
    )

    # Handle single subplot case
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, matrix in enumerate(matrices):
        ax = axes[idx]

        sns.heatmap(
            matrix,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            cmap="RdBu_r",
            square=True,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )

        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
        else:
            ax.set_title(f"Sample {idx}", fontsize=10)

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.00)

    plt.tight_layout()

    return fig, axes
