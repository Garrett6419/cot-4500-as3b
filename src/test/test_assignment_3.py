import unittest
import sys
import os
import numpy as np

# Adjust the path to point to the 'main' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../main')))

from assignment_3 import (
    gaussian_elimination,
    lu_factorization,
    is_diagonally_dominant,
    is_positive_definite
)

class TestAssignment3(unittest.TestCase):

    def test_gaussian_elimination(self):
        A = np.array([[2, 1, -1], [1, 3, 2], [1, -1, 2]], dtype=float)
        b = np.array([8, 13, 3], dtype=float)
        expected_result = np.array([3.3, 2.5, 1.1], dtype=float)
        result = gaussian_elimination(A.copy(), b.copy())
        np.testing.assert_array_almost_equal(result, expected_result, decimal=5)

    def test_lu_factorization(self):
        # Test LU factorization
        matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=float)
        L, U = lu_factorization(matrix)
        
        # Verify these expected values based on your LU factorization logic
        expected_L = np.array([[1, 0, 0, 0], [2, 1, 0, 0], [3, 4, 1, 0], [-1, -3, 0, 1]], dtype=float)
        expected_U = np.array([[1, 1, 0, 3], [0, -1, -1, -5], [0, 0, 3, 13], [0, 0, 0, -13]], dtype=float)
        
        np.testing.assert_array_almost_equal(L, expected_L, decimal=5)
        np.testing.assert_array_almost_equal(U, expected_U, decimal=5)

    def test_is_diagonally_dominant(self):
        diag_matrix = np.array([[3,1,1], [2,5,0],[-1,2,4]], dtype=float)
        self.assertTrue(is_diagonally_dominant(diag_matrix))

        non_diag_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        self.assertFalse(is_diagonally_dominant(non_diag_matrix))

    def test_is_positive_definite(self):
        # Test positive definiteness
        pos_def_matrix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]], dtype=float)
        self.assertTrue(is_positive_definite(pos_def_matrix))

        non_pos_def_matrix = np.array([[1, 2], [2, 1]], dtype=float)
        self.assertFalse(is_positive_definite(non_pos_def_matrix))

if __name__ == '__main__':
    unittest.main()