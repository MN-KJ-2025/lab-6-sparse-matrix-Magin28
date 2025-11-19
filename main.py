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
    
    if len(A.shape) != 2:
        return None
    
    if A.shape[0] != A.shape[1]:
        return None

    diag = np.diagonal(A)

    for i in range(0, A.shape[0]):
        if not np.abs(diag[i]) > np.sum(np.abs(A[i, :])) - np.abs(diag[i]):
            return False
    return True


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
