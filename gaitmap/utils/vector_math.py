"""A set of helper functions for find_shortest_rotation."""
from typing import Union

import numpy as np
from numpy.linalg import norm


def _row_wise_dot(v1, v2):
    """Calculate row wise dot product of two vectors."""
    if len(v1.shape) == 1:
        ax = 0
    else:
        ax = 1
    return np.sum(v1 * v2, axis=ax)


def inverse(v: np.array) -> np.array:
    """Find inverse quaternion.

    Parameters
    ----------
    v
        vector with shape (4,)

    Returns
    -------
    inverse of quaternion

    Examples
    --------
     >>> inverse(np.array([0, 1, 0, 0])))
    array([0, -1, 0, 0 ])

    """
    s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    return np.array([v[0] / s, -v[1] / s, -v[2] / s, -v[3] / s])


def inner_product(v1: np.array, v2: np.array) -> np.array:
    """Find Product of two quaternions.

    Parameters
    ----------
    v1
        vector with shape (4,)
    v2
        vector with shape (4,)

    Returns
    -------
    product of two quaternions

    Examples
    --------
    >>> inner_product(np.array([0, 1, 0, 0]), np.array([1, 0, 0, 0]))
    array([0, 1, 0, 0 ])

    """
    return np.array(
        [
            v1[0] * v2[0] - v1[1] * v2[1] - v1[2] * v2[2] - v1[3] * v2[3],
            v1[0] * v2[1] + v1[1] * v2[0] + v1[2] * v2[3] - v1[3] * v2[2],
            v1[0] * v2[2] - v1[1] * v2[3] + v1[2] * v2[0] + v1[3] * v2[1],
            v1[0] * v2[3] + v1[1] * v2[2] - v1[2] * v2[1] + v1[3] * v2[0],
        ]
    )


def is_almost_parallel_or_antiparallel(
    v1: np.ndarray, v2: np.ndarray, rtol: float = 1.0e-5, atol: float = 1.0e-8
) -> Union[bool, np.ndarray]:
    """Check if two vectors are either parallel or antiparallel.

    Parameters
    ----------
    v1 : vector with shape (3,) or array of vectors
        axis ([x, y ,z]) or array of axis
    v2 : vector with shape (3,) or array of vectors
        axis ([x, y ,z]) or array of axis
    rtol : float
        The relative tolerance parameter
    atol : float
        The absolute tolerance parameter

    Returns
    -------
    bool or array of bool values with len n

    Examples
    --------
    two vectors each of shape (3,)

    >>> is_almost_parallel_or_antiparallel(np.array([0, 0, 1]), np.array([0, 0, 1]))
    True
    >>> is_almost_parallel_or_antiparallel(np.array([0, 0, 1]), np.array([0, 1, 0]))
    False

    array of vectors

    >>> is_almost_parallel_or_antiparallel(np.array([[0, 0, 1],[0,1,0]]), np.array([[0, 0, 2],[1,0,0]]))
    array([True,False])

    """
    return np.isclose(np.abs(_row_wise_dot(normalize(v1), normalize(v2))), 1, rtol=rtol, atol=atol)


def normalize(v: np.ndarray) -> np.ndarray:
    """Simply normalize a vector.

    If a 2D array is provided, each row is considered a vector, which is normalized independently.

    Parameters
    ----------
    v : array with shape (3,) or (n, 3)
         vector or array of vectors

    Returns
    -------
    normalized vector or  array of normalized vectors

    Examples
    --------
    1D array

    >>> normalize(np.array([0, 0, 2]))
    array([0, 0, 1])

    2D array

    >>> normalize(np.array([[2, 0, 0],[2, 0, 0]]))
    array([[1, 0, 0],[1, 0, 0]])

    """
    v = np.array(v)
    if not v.any():
        raise ValueError("one element at least should have value other than 0")
    if len(v.shape) == 1:
        ax = 0
    else:
        ax = 1
    return (v.T / norm(v, axis=ax)).T


def find_random_orthogonal(v: np.ndarray) -> np.ndarray:
    """Find a unitvector in the orthogonal plane to v.

    Parameters
    ----------
    v : vector with shape (3,)
         axis ([x, y ,z])

    Returns
    -------
    vector which is either crossproduct with [0,1,0] or [1,0,0].

    Examples
    --------
    two vectors each of shape (3,)

    >>> find_random_orthogonal(np.array([1, 0, 0]))
    array([0, 0, 1])

    """
    if is_almost_parallel_or_antiparallel(v, np.array([1.0, 0, 0])):
        result = np.cross(v, [0, 1, 0])
    else:
        result = np.cross(v, [1, 0, 0])
    return normalize(result)


def find_orthogonal(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return an orthogonal vector to 2 vectors.

    Parameters
    ----------
    v1 : vector with shape (3,)
         axis ([x, y ,z])
    v2 : vector with shape (3,)
         axis ([x, y ,z])

    Returns
    -------
        Returns the cross product of the two if they are not equal.

        Returns a random vector in the perpendicular plane if they are either parallel or antiparallel.
        (see :func:`find_random_orthogonal`

    Examples
    --------
    >>> find_orthogonal(np.array([1, 0, 0]),np.array([-1, 0, 0]))
    array([0, 0, -1])

    """
    if v1.ndim > 1 or v2.ndim > 1:
        raise ValueError("v1 and v2 need to be at max 1D (currently {}D and {}D".format(v1.ndim, v2.ndim))
    if is_almost_parallel_or_antiparallel(v1, v2):
        return find_random_orthogonal(v1)
    return normalize(np.cross(v1, v2))


def find_unsigned_3d_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Find the angle (in rad) between two  3D vectors.

    Parameters
    ----------
    v1 : vector with shape (3,)  or array of vectors
        axis ([x, y ,z]) or array of axis
    v2 : vector with shape (3,) or array of vectors
        axis ([x, y ,z]) or array of axis

    Returns
    -------
        angle or array of angles between two vectors

    Examples
    --------
    two vectors: 1D

    >>> find_unsigned_3d_angle(np.array([-1, 0, 0]), np.array([-1, 0, 0]))
    0

    two vectors: 2D

    >>> find_unsigned_3d_angle(np.array([[-1, 0, 0],[-1, 0, 0]]), np.array([[-1, 0, 0],[-1, 0, 0]]))
    array([0,0])

    """
    v1 = np.array(v1)
    v1 = normalize(v1)
    v2 = np.array(v2)
    v2 = normalize(v2)
    if len(v1.shape) == 1:
        ax = 0
    else:
        ax = 1
    return np.arccos(_row_wise_dot(v1, v2) / (norm(v1, axis=ax) * norm(v2, axis=ax)))
