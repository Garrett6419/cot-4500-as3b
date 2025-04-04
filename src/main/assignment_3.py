import numpy as np  # type: ignore

def gaussian_elimination(A, b):
    n = len(b)
    # Forward elimination
    for i in range(n):
        # Make the diagonal contain all non-zero elements
        for j in range(i + 1, n):
            if A[j][i] != 0:
                factor = A[j][i] / A[i][i]
                A[j] = A[j] - factor * A[i]
                b[j] = b[j] - factor * b[i]

    # Backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i + 1:], x[i + 1:])) / A[i][i]
    return x

def lu_factorization(matrix):
    n = matrix.shape[0]
    L = np.zeros_like(matrix)
    U = np.zeros_like(matrix)

    for i in range(n):
        L[i][i] = 1  # Diagonal elements of L are 1
        for j in range(i, n):
            U[i][j] = matrix[i][j] - np.dot(L[i][:i], U[:i][j])
        for j in range(i + 1, n):
            L[j][i] = (matrix[j][i] - np.dot(L[j][:i], U[:i][i])) / U[i][i]
    
    return L, U

def is_diagonally_dominant(matrix):
    n = matrix.shape[0]
    for i in range(n):
        sum_row = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if abs(matrix[i][i]) < sum_row:
            return False
    return True

def is_positive_definite(matrix):
    try:
        # A matrix is positive definite if all its leading principal minors are positive
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
    result_gaussian = gaussian_elimination(A.copy(), b.copy())
    print("Gaussian Elimination Result:", result_gaussian)

    # Task 2: LU Factorization
    matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=float)
    L, U = lu_factorization(matrix)
    print("L Matrix:\n", L)
    print("U Matrix:\n", U)
    print("Determinant:", np.linalg.det(matrix))

    # Task 3: Diagonal Dominance Check
    diag_matrix = np.array([[9, 0, 5, 2], [1, 3, 9, 1], [2, 1, 0, 1], [7, 2, 3, 4]], dtype=float)
    print("Is Diagonally Dominant:", is_diagonally_dominant(diag_matrix))

    # Task 4: Positive Definiteness Check
    pos_def_matrix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]], dtype=float)
    print("Is Positive Definite:", is_positive_definite(pos_def_matrix))