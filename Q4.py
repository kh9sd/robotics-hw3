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

"""
q4_b answer

Using SVD to find a transformation requires that the points in the respective matrices
are corresponding. In the case of noisy data, while this might add a bit of error into
the result, the general assumption of corresponding points still holds. However, shuffling
the points order completely destroys that, so the algorithm fails
"""

def closest_neighbor_from_to(from_pc: np.ndarray, to_pc: np.ndarray) -> np.ndarray:
    closest_correspondences_list = []

    for pt in from_pc:
        assert pt.shape == (3,)

        pt_mapped = np.tile(pt, (to_pc.shape[0], 1))
        assert pt_mapped.shape == to_pc.shape

        distance_array = np.linalg.norm(to_pc-pt_mapped, axis=1)
        assert distance_array.shape == (to_pc.shape[0], )

        closest_pt_index = np.argmin(distance_array)

        closest_correspondences_list.append(to_pc[closest_pt_index])

    result = np.array(closest_correspondences_list)
    assert result.shape == from_pc.shape

    return result


def pad_for_homogeneous_trans(A: np.ndarray) -> np.ndarray:
    assert A.ndim == 2
    assert A.shape[1] == 3

    current = np.ones((A.shape[0], 4))
    current[:, :3] = A
    assert current.shape == (A.shape[0], 4)

    return current


def apply_homo_trans(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert T.shape == (4,4)
    assert points.shape[1] == 3

    padded_points = pad_for_homogeneous_trans(points)
    assert padded_points.shape == (points.shape[0], 4)

    result = T @ padded_points.T
    assert result.shape == (4, points.shape[0])

    final_result = result.T
    assert final_result.shape == (points.shape[0], 4)
    final_result = final_result[:, 0:3]
    assert final_result.shape == points.shape

    return final_result


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

    ITERATION_LIMIT = 100
    current, TARGET = M, D

    cur_trans = np.eye(4)

    for i in range(ITERATION_LIMIT):
        print(i)

        closest_correspondence_matrix = closest_neighbor_from_to(current, TARGET)
        assert closest_correspondence_matrix.shape == current.shape
        next_trans = q4_a(current, closest_correspondence_matrix)

        cur_trans = next_trans @ cur_trans
        assert cur_trans.shape == (4,4)

        current = apply_homo_trans(next_trans, current)
        assert current.shape == (M.shape[0], 3)

    return cur_trans
