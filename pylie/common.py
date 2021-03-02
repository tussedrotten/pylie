import numpy as np


def to_rotation_matrix(R):
    """Fits an arbitrary nxn matrix to the closest element on SO(n)

    See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    :param R: An arbitrary nxn matrix
    :return: The closest valid nxn rotation matrix
    """
    u, s, v = np.linalg.svd(R)
    R = u @ v

    if np.linalg.det(R) < 0:
        s = np.ones(s.shape)
        s[-1] = -1.0
        R = u @ np.diag(s) @ v

    return R
