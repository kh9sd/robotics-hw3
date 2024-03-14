from typing import Tuple
import numpy as np
import utils


def q4_a(M: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Find transformation T such that D = T @ M. This assumes that M and D are
    corresponding (i.e. M[i] and D[i] correspond to same point)

    Attributes
    ----------
    M : np.ndarray
        Nx3 matrix of points
    D : np.ndarray
        Nx3 matrix of points

    Returns
    -------
    T : np.ndarray
        4x4 homogenous transformation matrix

    Hint
    ----
    use `np.linalg.svd` to perform singular value decomposition
    '''
    center_of_M = np.sum(M, axis=0)
    center_of_M /= M.shape[0]
    assert center_of_M.shape == (3,)
    center_M_tile = np.tile(center_of_M, (M.shape[0], 1))
    centered_M = M - center_M_tile

    center_of_D = np.sum(D, axis=0)
    center_of_D /= D.shape[0]
    assert center_of_D.shape == (3,)
    center_D_tile = np.tile(center_of_D, (D.shape[0], 1))
    centered_D = D - center_D_tile
    

    W = np.zeros((3, 3))
    for centered_M_pt, centered_D_pt in zip(centered_M, centered_D):
        centered_M_pt = centered_M_pt[..., np.newaxis]
        centered_D_pt = centered_D_pt[..., np.newaxis]
        assert centered_M_pt.shape == (3,1)
        assert centered_D_pt.shape == (3,1)

        W += centered_D_pt @ centered_M_pt.T

    U, S, V_trans = np.linalg.svd(W, full_matrices=True, compute_uv=True)
    assert U.shape == (3,3)
    assert V_trans.shape == (3,3)

    R = U @ V_trans
    assert R.shape == (3,3)
    trans = center_of_D - R @ center_of_M

    T = np.zeros((4,4))

    T[0:3, 0:3] = R
    T[0:3, 3] = trans
    T[3, 3] = 1

    # T = np.eye(4)
    return T


def q4_c(M: np.ndarray, D: np.ndarray) -> np.ndarray:
    '''
    Solves iterative closest point (ICP) to generate transformation T to best
    align the points clouds: D = T @ M

    Attributes
    ----------
    M : np.ndarray
        Nx3 matrix of points
    D : np.ndarray
        Nx3 matrix of points

    Returns
    -------
    T : np.ndarray
        4x4 homogenous transformation matrix

    Hint
    ----
    you should make use of the function `q4_a`
    '''

    ### Enter code below
    T = np.eye(4)
    return T
