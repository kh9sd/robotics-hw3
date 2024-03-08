from typing import Tuple
import numpy as np

import utils


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
    center = np.array([0,1,0])
    normal = np.array([0,0,1])
    return normal, center
