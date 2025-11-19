# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp
from scipy.sparse import issparse

def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca 
            weryfikacji.
    
    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana, 
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None
    
    if A.ndim !=2:
        return None
    
    if A.shape[0] != A.shape[1]:
        return None

    if issparse(A):
        A = A.toarray()

    # Diagonal elements
    diag_elements = np.abs(np.diagonal(A))

    # Sum of absolute values of non-diagonal elements in each row
    row_sums = np.sum(np.abs(A), axis=1) - diag_elements

    # Check if diagonal elements are greater than the sum of non-diagonal elements
    return np.all(diag_elements > row_sums)



def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:

    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None

    if A.shape[1] != x.shape[0]:
        return None
    if A.shape[0] != b.shape[0]:
        return None
    
    if x.ndim != 1 or b.ndim != 1:
        return None


    Ax = np.dot(A, x)

    residuum = b - Ax
    
    norma = np.linalg.norm(residuum)
    
    return norma
