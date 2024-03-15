from typing import Tuple
import numpy as np
import random
import math

def get_plane_projection_trans(axis):
    assert axis.shape == (3,)

    axis_new_axis = axis[..., np.newaxis]
    assert axis_new_axis.shape == (3,1)

    result = axis_new_axis @ axis_new_axis.T
    assert result.shape == (3,3)

    return np.eye(3) - result

def q3(P: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    Localize a cylinder in the point cloud. Given a point cloud as
    input, this function should locate the position, orientation,
    and radius of the cylinder

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting 100 points in 3D space
    N : np.ndarray
        Nx3 matrix denoting normals of pointcloud

    Returns
    -------
    center : np.ndarray
        array of shape (3,) denoting cylinder center
    axis : np.ndarray
        array of shape (3,) pointing along cylinder axis
    radius : float
        scalar radius of cylinder
    '''

    best_axis = None
    best_radius = None
    best_center = None
    best_inlier_count = -math.inf

    ### Enter code below
    for i in range(5000):
        print(i)
        random_idx = np.random.choice(N.shape[0], 2, replace=False)
        random_normals = N[random_idx, :]
        assert random_normals.shape == (2,3)

        cylinder_axis = np.cross(random_normals[0], random_normals[1])
        cylinder_axis /= np.linalg.norm(cylinder_axis)
        assert cylinder_axis.shape == (3,)

        radius = random.uniform(0.05, 0.10)
        # finding center
        random_sample_idx = random_idx[0]
        radius_vector = N[random_sample_idx] / np.linalg.norm(N[random_sample_idx]) * radius
        assert radius_vector.shape == (3,)
        cylinder_center = P[random_sample_idx] + radius_vector

        center_mapped = np.tile(cylinder_center, (P.shape[0], 1))
        assert center_mapped.shape == P.shape

        centered_pts = P - center_mapped

        projection_trans = get_plane_projection_trans(cylinder_axis)

        projected_pts = (projection_trans @ centered_pts.T).T
        assert projected_pts.shape == P.shape

        distance_array = np.linalg.norm(projected_pts, axis=1)
        assert distance_array.shape == (P.shape[0], )

        inliers_count = 0
        for dist in distance_array:
            if math.isclose(dist, radius, abs_tol=0.001):
                inliers_count += 1

        if inliers_count > best_inlier_count:
            best_inlier_count = inliers_count
            best_axis = cylinder_axis
            best_center = cylinder_center
            best_radius = radius

    assert best_center is not None
    assert best_axis is not None
    assert best_radius is not None

    return best_center, best_axis, best_radius
