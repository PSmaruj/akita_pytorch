"""
Utility functions for Akita v2 analysis and visualization.

This package contains shared utilities used across notebooks and scripts.
"""

from .analysis_utils import (
    compare_datasets,
    find_all_models,
    load_loss_file,
    plot_all_models,
    plot_dataset_comparison,
    plot_single_model,
    print_summary_statistics,
)
from .visualization_utils import (
    plot_comparison,
    plot_contact_map,
    plot_matrix_grid,
    set_diag,
    upper_triu_to_matrix,
)
from .data_utils import (
    one_hot_encode_sequence, 
    process_hic_matrix, 
    upper_triangular_to_vector
)

__all__ = [
    # Visualization
    "plot_contact_map",
    "set_diag",
    "upper_triu_to_matrix",
    "plot_comparison",
    "plot_matrix_grid",
    # Analysis
    "load_loss_file",
    "find_all_models",
    "plot_single_model",
    "plot_all_models",
    "print_summary_statistics",
    "compare_datasets",
    "plot_dataset_comparison",
    # Data
    "one_hot_encode_sequence", 
    "process_hic_matrix", 
    "upper_triangular_to_vector"
]
