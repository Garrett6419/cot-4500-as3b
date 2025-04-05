import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    
    # Forward elimination
    for i in range(n):
        # Pivoting: Find the maximum element in the current column
        max_row_index = np.argmax(np.abs(A[i:n, i])) + i
        if A[max_row_index, i] == 0:
            raise ValueError("Matrix is singular and cannot be solved.")
        
        # Swap the current row with the max row
        A[[i, max_row_index]] = A[[max_row_index, i]]
        b[i], b[max_row_index] = b[max_row_index], b[i]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            # print(f"Iteration i={i}, j={j}, factor={factor}")  # Debugging print
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]
            # print("Matrix A after elimination:\n", A)  # Debugging print
            # print("Vector b after elimination:\n", b)  # Debugging print

    # Backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]
    
    # print("Final solution x:\n", x)  # Debugging print
    return x

def lu_factorization(matrix):
    n = matrix.shape[0]
    L = np.zeros((n, n))  # Initialize L with zeros
    U = np.zeros((n, n))  # Initialize U with zeros

    for i in range(n):
        L[i][i] = 1  # Set diagonal elements of L to 1
        for j in range(i, n):
            U[i][j] = matrix[i][j]  # Start with the original matrix values
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]  # Subtract the contributions from L

        for j in range(i + 1, n):
            L[j][i] = matrix[j][i]  # Start with the original matrix values
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]  # Subtract the contributions from U
            L[j][i] /= U[i][i]  # Normalize by the pivot element

    return L, U

def is_diagonally_dominant(matrix):
    n = matrix.shape[0]
    for i in range(n):
        diagonal_element = abs(matrix[i][i])
        sum_row = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        # print(f"Row {i}: Diagonal = {diagonal_element}, Sum of others = {sum_row}")  # Debug
        if diagonal_element < sum_row:
            return False
    return True

def is_positive_definite(matrix):
    try:
        for i in range(1, matrix.shape[0] + 1):
            if np.linalg.det(matrix[:i, :i]) <= 0:
                return False
        return True
    except np.linalg.LinAlgError:
        return False

# Example usage of the algorithms
if __name__ == "__main__":
    # Task 1: Gaussian Elimination
    A = np.array([[2, 1, -1], [1, 3, 2], [1, -1, 2]], dtype=float)
    b = np.array([8, 13, 3], dtype=float)
    gaussian_result = gaussian_elimination(A.copy(), b.copy())
    print("Gaussian Elimination Result:", gaussian_result)

    # Task 2: LU Factorization
    matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=float)
    L, U = lu_factorization(matrix)
    print("L Matrix:\n", L)
    print("U Matrix:\n", U)

    # Task 3: Check for Diagonal Dominance
    diag_matrix = np.array([[9, 0, 5, 2], [1, 3, 9, 1], [2, 1, 0, 1], [7, 2, 3, 4]], dtype=float)
    print("Is Diagonally Dominant:", is_diagonally_dominant(diag_matrix))

    # Task 4: Check for Positive Definiteness
    pos_def_matrix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]], dtype=float)
    print("Is Positive Definite:", is_positive_definite(pos_def_matrix))