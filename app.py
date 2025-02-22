import streamlit as st

# --- Matrix Operations without relying on inverse libraries ---

def parse_matrix(matrix_str):
    """
    Parses a string input into a 2D list (matrix).
    Each row should be on a new line with space-separated numbers.
    """
    rows = matrix_str.strip().split('\n')
    matrix = []
    for row in rows:
        row_values = row.strip().split()
        if row_values:
            matrix.append([float(x) for x in row_values])
    return matrix

def matrix_transpose(A):
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]

def matrix_multiply(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    result = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_add(A, B):
    m = len(A)
    n = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]

def matrix_identity(n):
    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I

def matrix_inverse(A):
    """
    Computes the inverse of a square matrix A using Gauss–Jordan elimination.
    The matrix is augmented with the identity matrix, and row operations are
    applied to transform A into the identity matrix; the augmented side then becomes A⁻¹.
    """
    n = len(A)
    # Create augmented matrix [A | I]
    aug = [row[:] + identity_row for row, identity_row in zip(A, matrix_identity(n))]
    
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            # Swap with a row below if pivot is nearly zero
            swap_row = None
            for j in range(i+1, n):
                if abs(aug[j][i]) > 1e-12:
                    swap_row = j
                    break
            if swap_row is None:
                raise ValueError("Matrix is singular and cannot be inverted.")
            aug[i], aug[swap_row] = aug[swap_row], aug[i]
            pivot = aug[i][i]
        # Normalize pivot row
        for j in range(2*n):
            aug[i][j] /= pivot
        # Eliminate pivot column entries in other rows
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2*n):
                    aug[k][j] -= factor * aug[i][j]
    # Extract inverse matrix from augmented matrix
    inv = [row[n:] for row in aug]
    return inv

def add_regularization(M, lambda_val):
    """
    Adds a small regularization term (λI) to the square matrix M.
    """
    n = len(M)
    I = matrix_identity(n)
    for i in range(n):
        I[i][i] *= lambda_val
    return matrix_add(M, I)

def pseudo_inverse(A):
    """
    Computes the pseudo inverse of matrix A.
    For a full column rank matrix (m >= n): A⁺ = (AᵀA)⁻¹ Aᵀ
    For a full row rank matrix (m < n): A⁺ = Aᵀ (AAᵀ)⁻¹
    If the inversion fails (e.g., matrix nearly singular), a small regularization term is added.
    """
    m = len(A)
    n = len(A[0])
    At = matrix_transpose(A)
    if m >= n:
        AtA = matrix_multiply(At, A)
        try:
            inv_AtA = matrix_inverse(AtA)
        except ValueError:
            # Regularize if singular
            AtA_reg = add_regularization(AtA, 1e-10)
            inv_AtA = matrix_inverse(AtA_reg)
        pseudo = matrix_multiply(inv_AtA, At)
    else:
        AAt = matrix_multiply(A, At)
        try:
            inv_AAt = matrix_inverse(AAt)
        except ValueError:
            AAt_reg = add_regularization(AAt, 1e-10)
            inv_AAt = matrix_inverse(AAt_reg)
        pseudo = matrix_multiply(At, inv_AAt)
    return pseudo

def matrix_to_string(matrix):
    """Converts a 2D list (matrix) into a formatted string."""
    return "\n".join(" ".join(f"{x:.4f}" for x in row) for row in matrix)

# --- Streamlit User Interface ---

st.set_page_config(page_title="Pseudo Inverse Calculator", layout="wide")

st.title("Pseudo Inverse Calculator")
st.write("Enter your matrix below. Each row should be on a new line with space-separated numbers.")

# Text area for matrix input
matrix_input = st.text_area("Matrix Input", height=150)

if st.button("Calculate Pseudo Inverse"):
    try:
        A = parse_matrix(matrix_input)
        if not A:
            st.error("Matrix input is empty.")
        else:
            # Check all rows have the same length
            ncols = len(A[0])
            for row in A:
                if len(row) != ncols:
                    raise ValueError("All rows must have the same number of numbers.")
            pseudo = pseudo_inverse(A)
            input_matrix_str = matrix_to_string(A)
            result_matrix_str = matrix_to_string(pseudo)
            
            st.subheader("Input Matrix")
            st.code(input_matrix_str, language="plaintext")
            
            st.subheader("Pseudo Inverse")
            st.code(result_matrix_str, language="plaintext")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Sidebar with Additional Information ---

st.sidebar.title("How does this work")
st.sidebar.markdown("""
#### Algorithm Details

This app computes the pseudo inverse using a method based on normal equations:

- **Full Column Rank (m ≥ n):**  
  *A⁺ = (AᵀA)⁻¹ Aᵀ*

- **Full Row Rank (m < n):**  
  *A⁺ = Aᵀ (AAᵀ)⁻¹*

If the matrix to be inverted is singular or nearly singular, a small regularization term is added.  
The matrix inverse is calculated using Gauss–Jordan elimination.
""")
