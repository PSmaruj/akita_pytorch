"""
Utility functions for Akita v2 analysis and visualization.

This package contains shared utilities used across notebooks and scripts.
"""

from .visualization_utils import (
    plot_contact_map,
    set_diag,
    upper_triu_to_matrix,
    plot_comparison,
    plot_matrix_grid
)

from .analysis_utils import (
    load_loss_file,
    find_all_models,
    plot_single_model,
    plot_all_models,
    print_summary_statistics,
    compare_datasets,
    plot_dataset_comparison
)

__all__ = [
    # Visualization
    'plot_contact_map',
    'set_diag',
    'upper_triu_to_matrix',
    'plot_comparison',
    'plot_matrix_grid',
    # Analysis
    'load_loss_file',
    'find_all_models',
    'plot_single_model',
    'plot_all_models',
    'print_summary_statistics',
    'compare_datasets',
    'plot_dataset_comparison'
]