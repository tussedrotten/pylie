import numpy as np

from pylie import SO3


def test_construct_identity_with_no_args():
    so3 = SO3()
    np.testing.assert_array_equal(so3.matrix, np.identity(3))


def test_construct_with_matrix():
    R = np.array([[0, 0, 1],
                  [1, 0, 0],
                  [0, 1, 0]])
    so3 = SO3(R)
    np.testing.assert_array_equal(so3.matrix, R)

    # Matrices not elements of SO(3) should be fitted to SO(3)
    R_noisy = R + 0.1 + np.random.rand(3)
    so3_fitted = SO3(R_noisy)
    R_fitted = so3_fitted.matrix

    assert np.any(np.not_equal(R_fitted, R_noisy))
    np.testing.assert_almost_equal(np.linalg.det(R_fitted), 1, 14)
    np.testing.assert_almost_equal((R_fitted.T @ R_fitted), np.identity(3), 14)


def test_construct_with_angle_axis():
    angle = 3 * np.pi / 4
    axis = np.array([[1, 1, -1]]) / np.sqrt(3)

    theta_vec = angle * axis
    so3 = SO3.from_angle_axis(angle, axis)

    np.testing.assert_array_equal(so3.matrix, SO3.Exp(theta_vec).matrix)


def test_construct_with_roll_pitch_yaw():
    np.testing.assert_almost_equal(SO3.from_roll_pitch_yaw(np.pi / 2, 0, 0).matrix, SO3.rot_x(np.pi / 2).matrix, 14)
    np.testing.assert_almost_equal(SO3.from_roll_pitch_yaw(0, np.pi / 2, 0).matrix, SO3.rot_y(np.pi / 2).matrix, 14)
    np.testing.assert_almost_equal(SO3.from_roll_pitch_yaw(0, 0, np.pi / 2).matrix, SO3.rot_z(np.pi / 2).matrix, 14)

    roll = np.pi
    pitch = np.pi / 2
    yaw = np.pi
    so3 = SO3.from_roll_pitch_yaw(roll, pitch, yaw)

    expected = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]])

    np.testing.assert_almost_equal(so3.matrix, expected, 14)


def test_hat_returns_skew_symmetric_matrix():
    theta_vec = np.array([[1, 2, 3]]).T
    theta_hat = SO3.hat(theta_vec)
    assert theta_hat[0, 0] == 0
    assert theta_hat[0, 1] == -theta_vec[2]
    assert theta_hat[0, 2] == theta_vec[1]
    assert theta_hat[1, 0] == theta_vec[2]
    assert theta_hat[1, 1] == 0
    assert theta_hat[1, 2] == -theta_vec[0]
    assert theta_hat[2, 0] == -theta_vec[1]
    assert theta_hat[2, 1] == theta_vec[0]
    assert theta_hat[2, 2] == 0


def test_vee_extracts_correct_vector_from_skew_symmetric_matrix():
    theta_vec = np.array([[3, 2, 1]]).T
    theta_hat = SO3.hat(theta_vec)
    theta_hat_vee = SO3.vee(theta_hat)
    np.testing.assert_array_equal(theta_hat_vee, theta_vec)


def test_exp_returns_correct_rotation():
    theta = 0.5 * np.pi
    u_vec = np.array([[0, 1, 0]]).T

    so3 = SO3.Exp(theta * u_vec)

    np.testing.assert_almost_equal(so3.matrix[:, 0:1], np.array([[0, 0, -1]]).T, 14)
    np.testing.assert_almost_equal(so3.matrix[:, 1:2], np.array([[0, 1, 0]]).T, 14)
    np.testing.assert_almost_equal(so3.matrix[:, 2:3], np.array([[1, 0, 0]]).T, 14)


def test_exp_for_vector_close_to_zero_norm_returns_identity():
    theta = 1e-14
    u_vec = np.array([[0, 1, 0]]).T

    so3 = SO3.Exp(theta * u_vec)

    np.testing.assert_array_equal(so3.matrix, np.identity(3))


def test_log_returns_correct_angle_axis():
    theta = np.pi / 4
    u_vec = np.array([[1, 0, -1]]).T / np.sqrt(2)

    so3 = SO3.Exp(theta * u_vec)
    theta_log, u_vec_log = so3.Log(True)

    np.testing.assert_almost_equal(theta_log, theta, 14)
    np.testing.assert_almost_equal(u_vec_log, u_vec, 14)


def test_log_returns_correct_theta_vec():
    theta = np.pi / 6
    u_vec = np.array([[-1, 1, 1]]).T / np.sqrt(3)
    theta_vec = theta * u_vec

    so3 = SO3.Exp(theta_vec)

    theta_vec_log = so3.Log()

    np.testing.assert_almost_equal(theta_vec_log, theta_vec, 14)


def test_inverse_returns_transposed():
    so3 = SO3.from_angle_axis(np.pi / 4, np.array([[1, 1, 0]]).T / np.sqrt(2))
    so3_inv = so3.inverse()

    np.testing.assert_array_equal(so3_inv.matrix, so3.matrix.T)
    np.testing.assert_array_equal(so3_inv.inverse().matrix, so3.matrix)


def test_action_on_vector_works():
    unit_x = np.array([[1, 0, 0]]).T
    unit_y = np.array([[0, 1, 0]]).T
    unit_z = np.array([[0, 0, 1]]).T
    so3 = SO3.from_angle_axis(np.pi / 2, unit_y)

    np.testing.assert_almost_equal(so3.action(unit_x), -unit_z, 14)


def test_action_on_vector_with_operator_works():
    unit_x = np.array([[1, 0, 0]]).T
    so3 = SO3.from_angle_axis(np.pi, np.array([[0, 0, -1]]).T)

    np.testing.assert_almost_equal(so3 * unit_x, -unit_x, 14)


def test_composition_with_identity_works():
    so3 = SO3.from_angle_axis(np.pi / 4, np.array([[0, 1, 0]]).T)

    comp_with_identity = so3.compose(SO3())
    comp_from_identity = SO3().compose(so3)

    np.testing.assert_almost_equal(comp_with_identity.matrix, so3.matrix, 14)
    np.testing.assert_almost_equal(comp_from_identity.matrix, so3.matrix, 14)


def test_composition_returns_correct_rotation():
    so3_90_x = SO3.from_angle_axis(np.pi / 2, np.array([[1, 0, 0]]).T)
    so3_90_y = SO3.from_angle_axis(np.pi / 2, np.array([[0, 1, 0]]).T)

    so3_comp = so3_90_y.compose(so3_90_x)

    expected = np.array([[0, 1, 0],
                         [0, 0, -1],
                         [-1, 0, 0]])

    np.testing.assert_almost_equal(so3_comp.matrix, expected, 14)


def test_composition_with_operator_works():
    so3_90_z = SO3.from_angle_axis(np.pi / 2, np.array([[0, 0, 1]]).T)
    so3_n90_x = SO3.from_angle_axis(np.pi / 2, np.array([[-1, 0, 0]]).T)

    so3_comp = so3_n90_x @ so3_90_z

    expected = np.array([[0, -1, 0],
                         [0, 0, 1],
                         [-1, 0, 0]])

    np.testing.assert_almost_equal(so3_comp.matrix, expected, 14)


def test_adjoint_returns_rotation_matrix():
    so3_1 = SO3()
    so3_2 = SO3.from_angle_axis(3 * np.pi / 4, np.array([[1, 0, 0]]).T)

    np.testing.assert_array_equal(so3_1.adjoint(), so3_1.matrix)
    np.testing.assert_array_equal(so3_2.adjoint(), so3_2.matrix)


def test_perturbation_by_adding_a_simple_rotation_works():
    ident = SO3()
    pert = ident.oplus(np.pi / 2 * np.array([[-1, 0, 0]]).T)
    expected = np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, -1, 0]])

    np.testing.assert_almost_equal(pert.matrix, expected, 14)


def test_perturbation_by_adding_a_simple_rotation_with_operator_works():
    ident = SO3()
    pert = ident + (np.pi / 2 * np.array([[0, 1, 0]]).T)
    expected = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]])

    np.testing.assert_almost_equal(pert.matrix, expected, 14)


def test_difference_for_simple_rotation_works():
    X = SO3()
    theta_vec = np.pi / 3 * np.array([[0, 1, 0]]).T
    Y = SO3.Exp(theta_vec)
    theta_vec_diff = Y.ominus(X)

    np.testing.assert_array_equal(theta_vec_diff, theta_vec)


def test_difference_for_simple_rotation_with_operator_works():
    X = SO3()
    theta_vec = 3 * np.pi / 4 * np.array([[1, 0, -1]]).T / np.sqrt(2)
    Y = SO3.Exp(theta_vec)
    theta_vec_diff = Y - X

    np.testing.assert_almost_equal(theta_vec_diff, theta_vec, 14)


def test_rot_x_works():
    rot_90_x = SO3.rot_x(0.5 * np.pi)
    expected_matrix = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])

    np.testing.assert_almost_equal(rot_90_x.matrix, expected_matrix, 14)


def test_rot_y_works():
    rot_90_y = SO3.rot_y(0.5 * np.pi)
    expected_matrix = np.array([[0, 0, 1],
                                [0, 1, 0],
                                [-1, 0, 0]])

    np.testing.assert_almost_equal(rot_90_y.matrix, expected_matrix, 14)


def test_rot_z_works():
    rot_90_z = SO3.rot_z(0.5 * np.pi)
    expected_matrix = np.array([[0, -1, 0],
                                [1, 0, 0],
                                [0, 0, 1]])

    np.testing.assert_almost_equal(rot_90_z.matrix, expected_matrix, 14)


def test_jacobian_inverse():
    X = SO3.from_angle_axis(np.pi / 3, np.array([[1, -1, -1]]).T / np.sqrt(3))

    J_inv = X.jac_inverse_X_wrt_X()

    # Jacobian should be -X.adjoint().
    np.testing.assert_array_equal(J_inv, -X.adjoint())

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.oplus(delta).inverse() - X.inverse().oplus(J_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 14)


def test_jacobian_composition_XY_wrt_X():
    X = SO3.from_angle_axis(np.pi / 4, np.array([[1, -1, 1]]).T / np.sqrt(3))
    Y = SO3.from_angle_axis(np.pi / 2, np.array([[1, 0, 1]]).T / np.sqrt(2))

    J_comp_X = X.jac_composition_XY_wrt_X(Y)

    # Jacobian should be Y.inverse().adjoint()
    np.testing.assert_array_equal(J_comp_X, Y.inverse().adjoint())

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.oplus(delta).compose(Y) - X.compose(Y).oplus(J_comp_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 14)


def test_jacobian_composition_XY_wrt_Y():
    X = SO3.from_angle_axis(np.pi / 4, np.array([[1, -1, 1]]).T / np.sqrt(3))
    Y = SO3.from_angle_axis(np.pi / 2, np.array([[1, 0, 1]]).T / np.sqrt(2))

    J_comp_Y = X.jac_composition_XY_wrt_Y()

    # Jacobian should be identity
    np.testing.assert_array_equal(J_comp_Y, np.identity(3))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.compose(Y.oplus(delta)) - X.compose(Y).oplus(J_comp_Y @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 14)


def test_jacobian_action_Xx_wrt_X():
    X = SO3.from_angle_axis(3 * np.pi / 4, np.array([[1, -1, 1]]).T / np.sqrt(3))
    x = np.array([[1, 2, 3]]).T

    J_action_X = X.jac_action_Xx_wrt_X(x)

    # Jacobian should be -X.matrix * SO3.hat(x).
    np.testing.assert_array_equal(J_action_X, -X.matrix @ SO3.hat(x))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.oplus(delta).action(x) - (X.action(x) + J_action_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 5)


def test_jacobian_action_Xx_wrt_x():
    X = SO3.from_angle_axis(3 * np.pi / 4, np.array([[1, -1, 1]]).T / np.sqrt(3))
    x = np.array([[1, 2, 3]]).T

    J_action_x = X.jac_action_Xx_wrt_x()

    # Jacobian should be X.matrix.
    np.testing.assert_array_equal(J_action_x, X.matrix)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.action(x + delta) - (X.action(x) + J_action_x @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 14)


def test_jacobian_right():
    theta_vec = 3 * np.pi / 4 * np.array([[1, -1, 1]]).T / np.sqrt(3)

    J_r = SO3.jac_right(theta_vec)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = SO3.Exp(theta_vec + delta) - (SO3.Exp(theta_vec) + J_r @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 5)


def test_jacobian_left():
    theta_vec = np.pi / 4 * np.array([[-1, -1, 1]]).T / np.sqrt(3)

    # Should have J_l(theta_vec) == J_r(-theta_vec).
    np.testing.assert_almost_equal(SO3.jac_left(theta_vec), SO3.jac_right(-theta_vec), 14)


def test_jacobian_right_inverse():
    theta_vec = 3 * np.pi / 4 * np.array([[1, -1, 0]]).T / np.sqrt(2)
    X = SO3.Exp(theta_vec)

    J_r_inv = SO3.jac_right_inverse(theta_vec)

    # Should have J_l * J_r_inv = Exp(theta_vec).adjoint().
    J_l = SO3.jac_left(theta_vec)
    np.testing.assert_almost_equal(J_l @ J_r_inv, SO3.Exp(theta_vec).adjoint(), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.oplus(delta).Log() - (X.Log() + J_r_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 5)


def test_jacobian_left_inverse():
    theta_vec = np.pi / 4 * np.array([[-1, -1, 1]]).T / np.sqrt(3)

    # Should have J_l_inv(theta_vec) == J_r_inv(theta_vec).T
    np.testing.assert_almost_equal(SO3.jac_left_inverse(theta_vec), SO3.jac_right_inverse(theta_vec).T, 14)


def test_jacobian_X_oplus_tau_wrt_X():
    X = SO3.from_angle_axis(3 * np.pi / 4, np.array([[1, -1, 1]]).T / np.sqrt(3))
    theta_vec = np.pi / 4 * np.array([[-1, -1, 1]]).T / np.sqrt(3)

    J_oplus_X = X.jac_X_oplus_tau_wrt_X(theta_vec)

    # Should be Exp(tau).adjoint().inverse()
    np.testing.assert_almost_equal(J_oplus_X, np.linalg.inv(SO3.Exp(theta_vec).adjoint()), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.oplus(delta).oplus(theta_vec) - X.oplus(theta_vec).oplus(J_oplus_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 14)


def test_jacobian_X_oplus_tau_wrt_tau():
    X = SO3.from_angle_axis(np.pi / 4, np.array([[-1, -1, -1]]).T / np.sqrt(3))
    theta_vec = np.pi / 3 * np.array([[-1, -1, 1]]).T / np.sqrt(3)

    J_oplus_tau = X.jac_X_oplus_tau_wrt_tau(theta_vec)

    # Should be J_r.
    np.testing.assert_equal(J_oplus_tau, X.jac_right(theta_vec))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.oplus(theta_vec + delta) - X.oplus(theta_vec).oplus(J_oplus_tau @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 6)


def test_jacobian_Y_ominus_X_wrt_X():
    X = SO3.from_angle_axis(np.pi / 4, np.array([[1, -1, 1]]).T / np.sqrt(3))
    Y = SO3.from_angle_axis(np.pi / 2, np.array([[1, 0, 1]]).T / np.sqrt(2))

    J_ominus_X = Y.jac_Y_ominus_X_wrt_X(X)

    # Should be -J_l_inv.
    np.testing.assert_equal(J_ominus_X, -SO3.jac_left_inverse(Y - X))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = Y.ominus(X.oplus(delta)) - (Y.ominus(X) + (J_ominus_X @ delta))
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 6)


def test_jacobian_Y_ominus_X_wrt_Y():
    X = SO3.from_angle_axis(np.pi / 4, np.array([[1, 1, 1]]).T / np.sqrt(3))
    Y = SO3.from_angle_axis(np.pi / 3, np.array([[1, 0, -1]]).T / np.sqrt(2))

    J_ominus_Y = Y.jac_Y_ominus_X_wrt_Y(X)

    # Should be J_r_inv.
    np.testing.assert_equal(J_ominus_Y, SO3.jac_right_inverse(Y - X))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = Y.oplus(delta).ominus(X) - (Y.ominus(X) + (J_ominus_Y @ delta))
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 6)


def test_has_len_that_returns_correct_dimension():
    X = SO3()

    # Should have 6 DOF.
    np.testing.assert_equal(len(X), 3)


def test_string_representation_is_matrix():
    X = SO3()

    # Should be the same as the string representation of the matrix.
    np.testing.assert_equal(str(X), str(X.matrix))
