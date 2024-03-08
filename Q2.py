from typing import Tuple
import numpy as np

import random
import math


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
    best_inlier_count = -math.inf
    best = None

    for i in range(500):
        print(i)
        random_pt_idx = random.randrange(0, P.shape[0])

        random_pt = P[random_pt_idx]
        assert random_pt.shape == (3,)

        radius_len = random.uniform(0.05, 0.11)

        radius_vector = N[random_pt_idx]
        radius_vector /= np.linalg.norm(radius_vector) # normalize
        radius_vector *= radius_len
        assert radius_vector.shape == (3,)

        circle_center = random_pt - radius_vector
        assert circle_center.shape == (3,)
        inliers_count = 0

        # we using numpys vectorized implementations :flushed:
        circle_center_mapped = np.tile(circle_center, (P.shape[0], 1))
        assert circle_center_mapped.shape == P.shape

        distance_array = np.linalg.norm(P-circle_center_mapped, axis=1)
        assert distance_array.shape == (P.shape[0], )

        for dist in distance_array:
            if math.isclose(dist, radius_len, abs_tol=0.01):
                inliers_count += 1

        if inliers_count > best_inlier_count:
            best_inlier_count = inliers_count
            best = (circle_center, radius_len)

    assert best is not None
    return best
