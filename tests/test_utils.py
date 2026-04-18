"""
Unit tests for critical Akita utility functions.

Run with: pytest tests/test_utils.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

from training.training_utils import compute_loss
from utils.visualization_utils import set_diag, upper_triu_to_matrix

# =============================================================================
# Tests for set_diag
# =============================================================================


class TestSetDiag:
    """Test suite for set_diag function."""

    def test_main_diagonal(self):
        """Test setting the main diagonal (k=0)."""
        matrix = np.ones((5, 5))
        set_diag(matrix, 0, k=0)

        # Check main diagonal is 0
        assert np.all(np.diag(matrix) == 0), "Main diagonal should be 0"

        # Check off-diagonal elements are still 1
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert matrix[i, j] == 1, f"Off-diagonal element ({i},{j}) should be 1"

    def test_upper_diagonal(self):
        """Test setting an upper diagonal (k>0)."""
        matrix = np.ones((5, 5))
        set_diag(matrix, 0, k=1)

        # Check first upper diagonal is 0
        for i in range(4):
            assert matrix[i, i + 1] == 0, f"Element ({i},{i + 1}) should be 0"

        # Check main diagonal is still 1
        assert np.all(np.diag(matrix) == 1), "Main diagonal should still be 1"

    def test_lower_diagonal(self):
        """Test setting a lower diagonal (k<0)."""
        matrix = np.ones((5, 5))
        set_diag(matrix, 0, k=-1)

        # Check first lower diagonal is 0
        for i in range(1, 5):
            assert matrix[i, i - 1] == 0, f"Element ({i},{i - 1}) should be 0"

        # Check main diagonal is still 1
        assert np.all(np.diag(matrix) == 1), "Main diagonal should still be 1"

    def test_multiple_diagonals(self):
        """Test setting multiple diagonals."""
        matrix = np.ones((5, 5))

        # Set main diagonal and adjacent diagonals
        set_diag(matrix, 0, k=-1)
        set_diag(matrix, 0, k=0)
        set_diag(matrix, 0, k=1)

        # Check that the tridiagonal band is 0
        for i in range(5):
            for j in range(5):
                if abs(i - j) <= 1:
                    assert matrix[i, j] == 0, f"Element ({i},{j}) should be 0"
                else:
                    assert matrix[i, j] == 1, f"Element ({i},{j}) should be 1"

    def test_out_of_bounds(self):
        """Test that out-of-bounds diagonals don't cause errors."""
        matrix = np.ones((3, 3))

        # These should not raise errors
        set_diag(matrix, 0, k=10)  # Way above
        set_diag(matrix, 0, k=-10)  # Way below

        # Matrix should be unchanged
        assert np.all(matrix == 1), "Matrix should be unchanged for out-of-bounds diagonals"

    def test_with_nan(self):
        """Test setting diagonals to NaN."""
        matrix = np.ones((4, 4))
        set_diag(matrix, np.nan, k=0)

        # Check main diagonal is NaN
        assert np.all(np.isnan(np.diag(matrix))), "Main diagonal should be NaN"

        # Check off-diagonal elements are still 1
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert matrix[i, j] == 1, f"Off-diagonal element ({i},{j}) should be 1"

    def test_rectangular_matrix(self):
        """Test with non-square matrix."""
        matrix = np.ones((3, 5))
        set_diag(matrix, 0, k=0)

        # Check main diagonal is 0 (min(3,5) = 3 elements)
        for i in range(3):
            assert matrix[i, i] == 0, f"Element ({i},{i}) should be 0"


# =============================================================================
# Tests for upper_triu_to_matrix
# =============================================================================


class TestUpperTriuToMatrix:
    """Test suite for upper_triu_to_matrix function."""

    def test_basic_reconstruction(self):
        """Test basic reconstruction of a simple matrix."""
        # Create a simple 5x5 matrix
        matrix_size = 5
        num_diags = 2

        # Create upper triangular indices
        triu_indices = np.triu_indices(matrix_size, num_diags)
        num_elements = len(triu_indices[0])

        # Create a vector with known values
        vector = np.arange(num_elements, dtype=np.float32)

        # Reconstruct matrix
        result = upper_triu_to_matrix(vector, matrix_len=matrix_size, num_diags=num_diags)

        # Check shape
        assert result.shape == (
            matrix_size,
            matrix_size,
        ), f"Expected shape ({matrix_size}, {matrix_size}), got {result.shape}"

        # Check symmetry
        np.testing.assert_array_equal(
            np.nan_to_num(result), np.nan_to_num(result.T), err_msg="Matrix should be symmetric"
        )

    def test_symmetry(self):
        """Test that reconstructed matrix is symmetric."""
        matrix_size = 10
        num_diags = 2

        # Create random upper triangular vector
        triu_indices = np.triu_indices(matrix_size, num_diags)
        num_elements = len(triu_indices[0])
        vector = np.random.randn(num_elements).astype(np.float32)

        # Reconstruct
        result = upper_triu_to_matrix(vector, matrix_len=matrix_size, num_diags=num_diags)

        # Check symmetry (ignoring NaN values)
        mask = ~np.isnan(result)
        assert np.allclose(
            result[mask], result.T[mask]
        ), "Matrix should be symmetric (excluding NaN)"

    def test_diagonal_nan(self):
        """Test that near-diagonal elements are set to NaN."""
        matrix_size = 10
        num_diags = 2

        # Create vector
        triu_indices = np.triu_indices(matrix_size, num_diags)
        num_elements = len(triu_indices[0])
        vector = np.ones(num_elements, dtype=np.float32)

        # Reconstruct
        result = upper_triu_to_matrix(vector, matrix_len=matrix_size, num_diags=num_diags)

        # Check that diagonals from -1 to 1 are NaN
        for k in range(-num_diags + 1, num_diags):
            diagonal = np.diag(result, k=k)
            assert np.all(np.isnan(diagonal)), f"Diagonal k={k} should be all NaN"

    def test_torch_tensor_input(self):
        """Test with PyTorch tensor input."""
        matrix_size = 8
        num_diags = 2

        # Create PyTorch tensor
        triu_indices = np.triu_indices(matrix_size, num_diags)
        num_elements = len(triu_indices[0])
        vector = torch.randn(num_elements)

        # Should not raise error
        result = upper_triu_to_matrix(vector, matrix_len=matrix_size, num_diags=num_diags)

        # Check it's numpy array
        assert isinstance(result, np.ndarray), "Output should be numpy array"

        # Check shape
        assert result.shape == (
            matrix_size,
            matrix_size,
        ), f"Expected shape ({matrix_size}, {matrix_size}), got {result.shape}"

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        matrix_size = 8
        num_diags = 2

        # Create numpy array
        triu_indices = np.triu_indices(matrix_size, num_diags)
        num_elements = len(triu_indices[0])
        vector = np.random.randn(num_elements).astype(np.float32)

        # Should not raise error
        result = upper_triu_to_matrix(vector, matrix_len=matrix_size, num_diags=num_diags)

        # Check shape
        assert result.shape == (
            matrix_size,
            matrix_size,
        ), f"Expected shape ({matrix_size}, {matrix_size}), got {result.shape}"

    def test_different_diagonal_offsets(self):
        """Test with different diagonal offset values."""
        matrix_size = 10

        for num_diags in [1, 2, 3, 5]:
            triu_indices = np.triu_indices(matrix_size, num_diags)
            num_elements = len(triu_indices[0])
            vector = np.ones(num_elements, dtype=np.float32)

            result = upper_triu_to_matrix(vector, matrix_len=matrix_size, num_diags=num_diags)

            # Check that correct number of diagonals are NaN
            for k in range(-num_diags + 1, num_diags):
                diagonal = np.diag(result, k=k)
                assert np.all(
                    np.isnan(diagonal)
                ), f"For num_diags={num_diags}, diagonal k={k} should be NaN"

    def test_realistic_hic_size(self):
        """Test with realistic Hi-C matrix size (512x512)."""
        matrix_size = 512
        num_diags = 2

        # Create vector
        triu_indices = np.triu_indices(matrix_size, num_diags)
        num_elements = len(triu_indices[0])
        vector = np.random.randn(num_elements).astype(np.float32)

        # Reconstruct
        result = upper_triu_to_matrix(vector, matrix_len=matrix_size, num_diags=num_diags)

        # Check shape
        assert result.shape == (
            matrix_size,
            matrix_size,
        ), f"Expected shape ({matrix_size}, {matrix_size}), got {result.shape}"

        # Check symmetry
        mask = ~np.isnan(result)
        assert np.allclose(result[mask], result.T[mask]), "Large matrix should be symmetric"

    def test_round_trip(self):
        """Test that the upper triangular values are preserved in the symmetric matrix."""
        matrix_size = 10
        num_diags = 2

        # Create original vector with known values
        triu_indices = np.triu_indices(matrix_size, num_diags)
        num_elements = len(triu_indices[0])
        original_vector = np.arange(num_elements, dtype=np.float32)

        # Reconstruct matrix
        matrix = upper_triu_to_matrix(original_vector, matrix_len=matrix_size, num_diags=num_diags)

        # The function places values in upper triangle, then adds the transpose
        # Since upper and lower triangles don't overlap (and diagonal is NaN),
        # the upper triangle values should be preserved as-is
        extracted_vector = matrix[triu_indices]

        # Values should match exactly (not doubled, since upper + lower don't overlap)
        np.testing.assert_array_almost_equal(
            original_vector,
            extracted_vector,
            decimal=5,
            err_msg="Upper triangular values should be preserved in the symmetric matrix",
        )

        # Also verify symmetry: check that lower triangle has the same values
        lower_triu_indices = (triu_indices[1], triu_indices[0])  # Swap row/col indices
        lower_values = matrix[lower_triu_indices]

        np.testing.assert_array_almost_equal(
            extracted_vector,
            lower_values,
            decimal=5,
            err_msg="Matrix should be symmetric: upper and lower triangles should match",
        )


# =============================================================================
# Tests for compute_loss
# =============================================================================


class TestComputeLoss:
    """Test suite for compute_loss function."""

    def test_no_nans(self):
        """Test loss computation with no NaN values."""
        output = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])

        loss = compute_loss(output, target)

        # Should compute MSE normally
        expected_loss = torch.nn.functional.mse_loss(output, target)

        assert torch.isclose(
            loss, expected_loss, atol=1e-6
        ), f"Expected {expected_loss}, got {loss}"

    def test_some_nans(self):
        """Test loss computation with some NaN values in target."""
        output = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[1.5, float("nan"), 3.5], [4.5, 5.5, float("nan")]])

        loss = compute_loss(output, target)

        # Should only compute MSE on non-NaN values
        valid_mask = ~torch.isnan(target)
        expected_loss = torch.nn.functional.mse_loss(output[valid_mask], target[valid_mask])

        assert torch.isclose(
            loss, expected_loss, atol=1e-6
        ), f"Expected {expected_loss}, got {loss}"

    def test_all_nans(self):
        """Test loss computation when all target values are NaN."""
        output = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = torch.tensor(
            [[float("nan"), float("nan"), float("nan")], [float("nan"), float("nan"), float("nan")]]
        )

        loss = compute_loss(output, target)

        # Should return 0 loss
        assert loss.item() == 0.0, "All NaN targets should return 0 loss"

    def test_loss_is_positive(self):
        """Test that loss is always non-negative."""
        output = torch.randn(10, 20)
        target = torch.randn(10, 20)

        loss = compute_loss(output, target)

        assert loss >= 0, "Loss should be non-negative"

    def test_perfect_prediction(self):
        """Test that identical output and target give zero loss."""
        output = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = output.clone()

        loss = compute_loss(output, target)

        assert torch.isclose(
            loss, torch.tensor(0.0), atol=1e-6
        ), "Perfect prediction should give zero loss"

    def test_device_compatibility(self):
        """Test that function works with tensors on different devices."""
        # CPU
        output_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_cpu = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

        loss_cpu = compute_loss(output_cpu, target_cpu)
        assert loss_cpu.device.type == "cpu", "Loss should be on CPU"

        # GPU (if available)
        if torch.cuda.is_available():
            output_gpu = output_cpu.cuda()
            target_gpu = target_cpu.cuda()

            loss_gpu = compute_loss(output_gpu, target_gpu)
            assert loss_gpu.device.type == "cuda", "Loss should be on GPU"

            # Values should be the same
            assert torch.isclose(
                loss_cpu, loss_gpu.cpu(), atol=1e-6
            ), "Loss should be same on CPU and GPU"

    def test_gradient_flow(self):
        """Test that gradients can flow through the loss."""
        output = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        target = torch.tensor([[1.5, 2.5, 3.5]])

        loss = compute_loss(output, target)
        loss.backward()

        # Check that gradients exist
        assert output.grad is not None, "Gradients should exist"
        assert not torch.all(output.grad == 0), "Gradients should be non-zero"

    def test_batch_dimension(self):
        """Test with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            output = torch.randn(batch_size, 100)
            target = torch.randn(batch_size, 100)

            loss = compute_loss(output, target)

            # Should return scalar
            assert loss.dim() == 0, "Loss should be scalar"
            assert loss >= 0, "Loss should be non-negative"

    def test_nan_mask_correctness(self):
        """Test that NaN masking is applied correctly."""
        output = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[1.0, float("nan"), 3.0, float("nan")]])

        loss = compute_loss(output, target)

        # Manually compute expected loss (only positions 0 and 2)
        valid_output = torch.tensor([1.0, 3.0])
        valid_target = torch.tensor([1.0, 3.0])
        expected_loss = torch.nn.functional.mse_loss(valid_output, valid_target)

        assert torch.isclose(
            loss, expected_loss, atol=1e-6
        ), f"Expected {expected_loss}, got {loss}"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    # Run with pytest if available, otherwise run basic checks
    try:
        import pytest

        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not installed. Running basic tests...")

        # Run a few basic tests
        test_set_diag = TestSetDiag()
        test_set_diag.test_main_diagonal()
        test_set_diag.test_upper_diagonal()
        print("✓ set_diag tests passed")

        test_upper_triu = TestUpperTriuToMatrix()
        test_upper_triu.test_basic_reconstruction()
        test_upper_triu.test_symmetry()
        print("✓ upper_triu_to_matrix tests passed")

        test_loss = TestComputeLoss()
        test_loss.test_no_nans()
        test_loss.test_some_nans()
        test_loss.test_all_nans()
        print("✓ compute_loss tests passed")

        print("\nAll basic tests passed! Install pytest for comprehensive testing.")
