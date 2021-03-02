from pylie.common import *
from pylie import SO3


def test_to_rotation_matrix_results_in_valid_rotation():
    # Test 2x2 matrix.
    A = np.random.rand(2, 2)
    R = to_rotation_matrix(A)
    np.testing.assert_almost_equal(R @ R.T, np.identity(2), 14)
    np.testing.assert_almost_equal(np.linalg.det(R), 1, 14)

    # Test 3x3 matrix.
    A = np.random.rand(3, 3)
    R = to_rotation_matrix(A)
    np.testing.assert_almost_equal(R @ R.T, np.identity(3), 14)
    np.testing.assert_almost_equal(np.linalg.det(R), 1, 14)


def test_to_rotation_matrix_results_in_close_rotation():
    angle = 0.5 * np.pi
    axis = np.array([[1, 0, 0]]).T
    R = SO3.Exp(angle * axis).matrix

    # Invalidate a valid rotation matrix by scaling it.
    R_scaled = 3 * R

    # Fit to SO(3).
    R_closest = to_rotation_matrix(R_scaled)

    # Result should be the same rotation matrix.
    np.testing.assert_almost_equal(R_closest, R, 14)

    # Perturb the rotation matrix with random noise.
    R_noisy = R + 0.01 * np.random.rand(3, 3)

    # Fit to SO(3)
    so3_closest = SO3(R_noisy)

    # Extract angle-axis representation.
    angle_closest, axis_closest = so3_closest.Log(True)

    # Result should be close to the same rotation.
    np.testing.assert_almost_equal(angle_closest, angle, 2)
    np.testing.assert_almost_equal(axis_closest, axis, 2)
