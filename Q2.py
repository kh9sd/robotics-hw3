from typing import Tuple
import numpy as np


def q2(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Localize a sphere in the point cloud. Given a point cloud as
    input, this function should locate the position and radius
    of a sphere

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space
    N : np.ndarray
        Nx3 matrix denoting normals of pointcloud

    Returns
    -------
    center : np.ndarray
        array of shape (3,) denoting sphere center
    radius : float
        scalar radius of sphere

    Hint
    ----
    use `utils.estimate_normals` to compute normals for point cloud
    '''

    ### Enter code below
    center = np.array([1,0.5,0.1])
    radius = 0.05
    return center, radius