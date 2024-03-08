from typing import Tuple
import numpy as np

import utils
import math


def q1_a(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit a least squares plane by taking the Eigen values and vectors
    of the sample covariance matrix

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space

    Returns
    -------
    normal : np.ndarray
        array of shape (3,) denoting surface normal of the fitting plane
    center : np.ndarray
        array of shape (3,) denoting center of the points
    '''

    ### Enter code below
   
    center = np.mean(P, axis=0)
    assert center.shape == (3,)

    covariance_matrix = np.zeros((3,3))

    for pt in P:
        assert pt.shape == (3,)

        pt_to_center_diff = np.expand_dims(pt - center, axis=0)
        assert pt_to_center_diff.shape == (1,3)

        # transpose first one, bc first should be a column vector
        covariance_matrix += pt_to_center_diff.T @ pt_to_center_diff
    
    eigenvals, eigenvecs = np.linalg.eig(covariance_matrix)
    # eigenvecs are the COLUMNS of the returned matrix, have to transpose to iterate properly
    _, normal = min(zip(eigenvals, eigenvecs.T), key=lambda pair: pair[0])

    return normal, center


def get_normal_from_3_pts(a: np.array, b: np.array, c: np.array):
    assert a.shape == (3,)
    assert b.shape == (3,)
    assert c.shape == (3,)

    return np.cross(a-b, c-b)


def is_inlier(sample_pt, plane_pt, plane_norm):
    # pt to pt on plane vector
    # normal vector
    # dot product gets us |a||b| cos
    # divide out length of normal vector
    diff_vec = sample_pt - plane_pt

    distance = abs(np.dot(diff_vec, plane_norm) / np.linalg.norm(plane_norm))

    return distance < 0.05


def q1_c(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit a plane using RANSAC

    Attributes
    ----------
    P : np.ndarray
        Nx3 matrix denoting points in 3D space

    Returns
    -------
    normal : np.ndarray
        array of shape (3,) denoting surface normal of the fitting plane
    center : np.ndarray
        array of shape (3,) denoting center of the points
    '''
    
    ### Enter code below
    best_inlier_count = -math.inf
    best = None

    for _ in range(100):
        random_3_pts = P[np.random.choice(P.shape[0], size=3, replace=False), :]
        assert random_3_pts.shape == (3,3)

        normal = get_normal_from_3_pts(random_3_pts[0], random_3_pts[1], random_3_pts[2])

        inliers_count = 0
        for pt in P:
            if is_inlier(pt, random_3_pts[0], normal):
                inliers_count += 1
        
        if inliers_count > best_inlier_count:
            best_inlier_count = inliers_count
            best = (normal, np.mean(random_3_pts, axis=0))

    assert best is not None
    return best

