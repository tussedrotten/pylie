import numpy as np

from pylie import SO3, SE3


def test_construct_identity_with_no_args():
    se3 = SE3()
    np.testing.assert_equal(se3.rotation.matrix, np.identity(3))
    np.testing.assert_equal(se3.translation, np.zeros((3, 1)))


def test_construct_with_tuple():
    so3 = SO3() + np.array([[0.1, -0.3, 0.2]]).T
    t = np.array([[1, 2, 3]]).T
    se3 = SE3((so3, t))

    np.testing.assert_equal(se3.rotation.matrix, so3.matrix)
    np.testing.assert_equal(se3.translation, t)


def test_construct_with_matrix():
    so3 = SO3.from_roll_pitch_yaw(np.pi / 10, np.pi / 4, -np.pi / 2)
    t = np.array([[1, 2, 3]]).T
    T = np.block([[so3.matrix, t],
                  [0, 0, 0, 1]])

    se3 = SE3.from_matrix(T)

    np.testing.assert_almost_equal(se3.rotation.matrix, so3.matrix, 14)
    np.testing.assert_equal(se3.translation, t)


def test_to_matrix():
    so3 = SO3.from_roll_pitch_yaw(np.pi / 4, np.pi / 2, 3 * np.pi / 2)
    t = np.array([[3, 2, 1]]).T
    se3 = SE3((so3, t))
    T = se3.to_matrix()

    np.testing.assert_equal(T[0:3, 0:3], so3.matrix)
    np.testing.assert_equal(T[0:3, 3:], t)
    np.testing.assert_equal(T[3:, :], np.array([[0, 0, 0, 1]]))


def test_to_tuple():
    so3 = SO3.from_roll_pitch_yaw(np.pi / 3, -np.pi / 2, -3 * np.pi / 2)
    t = np.array([[-1, 1, 3]]).T
    se3 = SE3((so3, t))
    pose_tuple = se3.to_tuple()

    np.testing.assert_equal(pose_tuple[0], so3.matrix)
    np.testing.assert_equal(pose_tuple[1], t)


def test_composition_with_identity():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 10, -np.pi / 3, 3 * np.pi / 4), np.array([[3, 2, 1]]).T))

    comp_with_identity = X.compose(SE3())
    comp_from_identity = SE3().compose(X)

    np.testing.assert_almost_equal(comp_with_identity.to_matrix(), X.to_matrix(), 14)
    np.testing.assert_almost_equal(comp_from_identity.to_matrix(), X.to_matrix(), 14)


def test_composition():
    X = SE3((SO3.rot_x(np.pi / 2), np.array([[1, 2, 3]]).T))
    Y = SE3((SO3.rot_z(np.pi / 2), np.array([[0, 1, 0]]).T))

    Z = X.compose(Y)

    rot_expected = SO3.from_angle_axis(2 * np.pi / 3, np.array([[1, -1, 1]]).T / np.sqrt(3))
    t_expected = np.array([[1, 2, 4]]).T

    np.testing.assert_almost_equal(Z.rotation.matrix, rot_expected.matrix, 14)
    np.testing.assert_almost_equal(Z.translation, t_expected, 14)


def test_composition_with_operator():
    X = SE3((SO3.rot_x(3 * np.pi / 2), np.array([[1, 2, 3]]).T))
    Y = SE3((SO3.rot_y(np.pi / 2), np.array([[-1, 0, 1]]).T))

    Z = X @ Y

    rot_expected = SO3.from_angle_axis(2 * np.pi / 3, np.array([[-1, 1, -1]]).T / np.sqrt(3))
    t_expected = np.array([[0, 3, 3]]).T

    np.testing.assert_almost_equal(Z.rotation.matrix, rot_expected.matrix, 14)
    np.testing.assert_almost_equal(Z.translation, t_expected, 14)


def test_inverse():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 4, np.pi / 2, 3 * np.pi / 2), np.array([[1, -2, 3]]).T))

    X_inv = X.inverse()

    np.testing.assert_equal(X_inv.rotation.matrix, X.rotation.inverse().matrix)
    np.testing.assert_equal(X_inv.translation, -(X.rotation.inverse() * X.translation))

    np.testing.assert_almost_equal((X @ X_inv).to_matrix(), np.identity(4))


def test_action_on_vectors():
    unit_x = np.array([[1, 0, 0]]).T
    unit_z = np.array([[0, 0, 1]]).T
    t = np.array([[3, 2, 1]]).T

    X = SE3((SO3.from_angle_axis(np.pi / 2, np.array([[0, 1, 0]]).T), t))

    np.testing.assert_almost_equal(X.action(unit_x), -unit_z + t, 14)


def test_action_on_vectors_with_operator():
    X = SE3((SO3.rot_x(3 * np.pi / 2), np.array([[1, 2, 3]]).T))
    x = np.array([[-1, 0, 1]]).T

    vec_expected = np.array([[0, 3, 3]]).T

    np.testing.assert_almost_equal(X * x, vec_expected, 14)


def test_hat():
    rho_vec = np.array([[1, 2, 3]]).T
    theta_vec = np.pi / 3 * np.array([[1, -1, 2]]).T / np.sqrt(6)
    xi_vec = np.vstack((rho_vec, theta_vec))

    xi_hat_expected = np.block([[SO3.hat(theta_vec), rho_vec],
                                [np.zeros((1, 4))]])

    np.testing.assert_equal(SE3.hat(xi_vec), xi_hat_expected)


def test_vee():
    rho_vec = np.array([[4, 3, 2]]).T
    theta_vec = 2 * np.pi / 3 * np.array([[2, 1, 1]]).T / np.sqrt(6)
    xi_vec = np.vstack((rho_vec, theta_vec))
    xi_hat = SE3.hat(xi_vec)

    np.testing.assert_equal(SE3.vee(xi_hat), xi_vec)


def test_adjoint():
    np.testing.assert_equal(SE3().adjoint(), np.identity(6))

    X = SE3((SO3.rot_x(3 * np.pi / 2), np.array([[1, 2, 3]]).T))
    Adj = X.adjoint()

    np.testing.assert_almost_equal(Adj[:3, :3], X.rotation.matrix, 14)
    np.testing.assert_almost_equal(Adj[3:, 3:], X.rotation.matrix, 14)
    np.testing.assert_almost_equal(Adj[:3, 3:], SO3.hat(X.translation) @ X.rotation.matrix, 14)


def test_log_with_no_rotation():
    X = SE3((SO3(), np.array([[-1, 2, -3]]).T))

    xi_vec = X.Log()

    # Translation part:
    np.testing.assert_equal(xi_vec[:3], X.translation)

    # Rotation part:
    np.testing.assert_equal(xi_vec[3:], np.zeros((3, 1)))


def test_exp_with_no_rotation():
    rho_vec = np.array([[1, 2, 3]]).T
    theta_vec = np.zeros((3, 1))
    xi_vec = np.vstack((rho_vec, theta_vec))

    X = SE3.Exp(xi_vec)

    np.testing.assert_equal(X.rotation.matrix, np.identity(3))
    np.testing.assert_equal(X.translation, rho_vec)


def test_log_of_exp():
    rho_vec = np.array([[1, 2, 3]]).T
    theta_vec = 2 * np.pi / 3 * np.array([[2, 1, -1]]).T / np.sqrt(6)
    xi_vec = np.vstack((rho_vec, theta_vec))

    X = SE3.Exp(xi_vec)

    np.testing.assert_almost_equal(X.Log(), xi_vec, 14)


def test_exp_of_log():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 10, -np.pi / 3, 3 * np.pi / 4), np.array([[3, 2, 1]]).T))
    xi_vec = X.Log()

    np.testing.assert_almost_equal(SE3.Exp(xi_vec).to_matrix(), X.to_matrix(), 14)


def test_ominus_with_oplus_diff():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 10, -np.pi / 3, 3 * np.pi / 4), np.array([[3, 2, 1]]).T))

    rho_vec = np.array([[4, 3, 2]]).T
    theta_vec = 2 * np.pi / 3 * np.array([[2, 1, 1]]).T / np.sqrt(6)
    xi_vec = np.vstack((rho_vec, theta_vec))

    Y = X.oplus(xi_vec)
    xi_vec_diff = Y.ominus(X)

    np.testing.assert_almost_equal(xi_vec_diff, xi_vec, 14)


def test_oplus_with_ominus_diff():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 10, -np.pi / 3, 3 * np.pi / 4), np.array([[3, 2, 1]]).T))
    Y = SE3((SO3.from_roll_pitch_yaw(np.pi / 7, np.pi / 3, 4 * np.pi / 6), np.array([[2, 1, 0]]).T))

    xi_vec_diff = Y.ominus(X)
    Y_from_X = X.oplus(xi_vec_diff)

    np.testing.assert_almost_equal(Y_from_X.to_matrix(), Y.to_matrix(), 14)


def test_ominus_with_oplus_diff_with_operators():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 7, np.pi / 3, 4 * np.pi / 6), np.array([[2, 1, 0]]).T))

    rho_vec = np.array([[1, 2, 3]]).T
    theta_vec = 2 * np.pi / 3 * np.array([[-2, 1, 1]]).T / np.sqrt(6)
    xi_vec = np.vstack((rho_vec, theta_vec))

    Y = X + xi_vec
    xi_vec_diff = Y - X

    np.testing.assert_almost_equal(xi_vec_diff, xi_vec, 14)


def test_jacobian_inverse():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 7, np.pi / 3, 4 * np.pi / 6), np.array([[2, 1, 0]]).T))

    J_inv = X.jac_inverse_X_wrt_X()

    # Jacobian should be -X.adjoint().
    np.testing.assert_equal(J_inv, -X.adjoint())

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = X.oplus(delta).inverse() - X.inverse().oplus(J_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 14)


def test_jacobian_action_Xx_wrt_X():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 8, np.pi / 2, 5 * np.pi / 6), np.array([[2, 1, 1]]).T))
    x = np.array([[1, 2, 3]]).T

    J_action_X = X.jac_action_Xx_wrt_X(x)

    # Jacobian should be [R,  - R*SO3.hat(x)].
    np.testing.assert_array_equal(J_action_X, np.block([[X.rotation.matrix, -(X.rotation.matrix @ SO3.hat(x))]]))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = X.oplus(delta).action(x) - (X.action(x) + J_action_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 5)


def test_jacobian_action_Xx_wrt_x():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 8, np.pi / 2, 5 * np.pi / 6), np.array([[2, 1, 1]]).T))
    x = np.array([[1, 2, 3]]).T

    J_action_x = X.jac_action_Xx_wrt_x()

    # Jacobian should be R.
    np.testing.assert_array_equal(J_action_x, X.rotation.matrix)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((3, 1))
    taylor_diff = X.action(x + delta) - (X.action(x) + J_action_x @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((3, 1)), 14)


def test_jacobian_composition_XY_wrt_X():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 10, -np.pi / 3, 3 * np.pi / 4), np.array([[3, 2, 1]]).T))
    Y = SE3((SO3.from_roll_pitch_yaw(np.pi / 7, np.pi / 3, 4 * np.pi / 6), np.array([[2, 1, 0]]).T))

    J_comp_X = X.jac_composition_XY_wrt_X(Y)

    # Jacobian should be Y.inverse().adjoint()
    np.testing.assert_almost_equal(J_comp_X, Y.inverse().adjoint(), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = X.oplus(delta).compose(Y) - X.compose(Y).oplus(J_comp_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 14)


def test_jacobian_composition_XY_wrt_Y():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 10, -np.pi / 3, 3 * np.pi / 4), np.array([[3, 2, 1]]).T))
    Y = SE3((SO3.from_roll_pitch_yaw(np.pi / 7, np.pi / 3, 4 * np.pi / 6), np.array([[2, 1, 0]]).T))

    J_comp_Y = X.jac_composition_XY_wrt_Y()

    # Jacobian should be identity
    np.testing.assert_array_equal(J_comp_Y, np.identity(6))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = X.compose(Y.oplus(delta)) - X.compose(Y).oplus(J_comp_Y @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 14)


def test_jacobian_right():
    rho_vec = np.array([[1, 2, 3]]).T
    theta_vec = 3 * np.pi / 4 * np.array([[1, -1, 1]]).T / np.sqrt(3)
    xi_vec = np.vstack((rho_vec, theta_vec))

    J_r = SE3.jac_right(xi_vec)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = SE3.Exp(xi_vec + delta) - (SE3.Exp(xi_vec) + J_r @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 5)


def test_jacobian_left():
    rho_vec = np.array([[2, 1, 2]]).T
    theta_vec = np.pi / 4 * np.array([[-1, -1, -1]]).T / np.sqrt(3)
    xi_vec = np.vstack((rho_vec, theta_vec))

    J_l = SE3.jac_left(xi_vec)

    # Should have J_l(xi_vec) == J_r(-xi_vec).
    np.testing.assert_almost_equal(J_l, SE3.jac_right(-xi_vec), 14)

    # Test the Jacobian numerically (using Exps and Logs, since left oplus and ominus have not been defined).
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = SE3.Log(SE3.Exp(xi_vec + delta) @ (SE3.Exp(J_l @ delta) @ SE3.Exp(xi_vec)).inverse())
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 5)


def test_jacobian_right_inverse():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 8, np.pi / 2, 5 * np.pi / 6), np.array([[2, 1, 1]]).T))
    xi_vec = X.Log()

    J_r_inv = SE3.jac_right_inverse(xi_vec)

    # Should have J_l * J_r_inv = Exp(xi_vec).adjoint().
    J_l = SE3.jac_left(xi_vec)
    np.testing.assert_almost_equal(J_l @ J_r_inv, SE3.Exp(xi_vec).adjoint(), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = X.oplus(delta).Log() - (X.Log() + J_r_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 5)


def test_jacobian_left_inverse():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 8, np.pi / 2, 5 * np.pi / 6), np.array([[2, 1, 1]]).T))
    xi_vec = X.Log()

    J_l_inv = SE3.jac_left_inverse(xi_vec)

    # Should have that left and right are block transposes of each other.
    J_r_inv = SE3.jac_right_inverse(xi_vec)
    np.testing.assert_almost_equal(J_l_inv[:3, :3], J_r_inv[:3, :3].T, 14)
    np.testing.assert_almost_equal(J_l_inv[:3, 3:], J_r_inv[:3, 3:].T, 14)
    np.testing.assert_almost_equal(J_l_inv[3:, 3:], J_r_inv[3:, 3:].T, 14)

    # Test the Jacobian numerically (using Exps and Logs, since left oplus and ominus have not been defined).
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = (SE3.Exp(delta) @ X).Log() - (X.Log() + J_l_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 5)


def test_jacobian_X_oplus_tau_wrt_X():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 10, -np.pi / 3, 3 * np.pi / 4), np.array([[3, 2, 1]]).T))
    rho_vec = np.array([[2, 1, 2]]).T
    theta_vec = np.pi / 4 * np.array([[-1, -1, -1]]).T / np.sqrt(3)
    xi_vec = np.vstack((rho_vec, theta_vec))

    J_oplus_X = X.jac_X_oplus_tau_wrt_X(xi_vec)

    # Should be Exp(tau).adjoint().inverse()
    np.testing.assert_almost_equal(J_oplus_X, np.linalg.inv(SE3.Exp(xi_vec).adjoint()), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = X.oplus(delta).oplus(xi_vec) - X.oplus(xi_vec).oplus(J_oplus_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 14)


def test_jacobian_X_oplus_tau_wrt_tau():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 8, np.pi / 2, 5 * np.pi / 6), np.array([[2, 1, 1]]).T))
    rho_vec = np.array([[2, 1, 2]]).T
    theta_vec = np.pi / 4 * np.array([[-1, -1, -1]]).T / np.sqrt(3)
    xi_vec = np.vstack((rho_vec, theta_vec))

    J_oplus_tau = X.jac_X_oplus_tau_wrt_tau(xi_vec)

    # Should be J_r.
    np.testing.assert_equal(J_oplus_tau, X.jac_right(xi_vec))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = X.oplus(xi_vec + delta) - X.oplus(xi_vec).oplus(J_oplus_tau @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 6)


def test_jacobian_Y_ominus_X_wrt_X():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 8, np.pi / 2, 5 * np.pi / 6), np.array([[2, 1, 1]]).T))
    Y = SE3((SO3.from_roll_pitch_yaw(np.pi / 7, np.pi / 3, 4 * np.pi / 6), np.array([[2, 1, 0]]).T))

    J_ominus_X = Y.jac_Y_ominus_X_wrt_X(X)

    # Should be -J_l_inv.
    np.testing.assert_equal(J_ominus_X, -SE3.jac_left_inverse(Y - X))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = Y.ominus(X.oplus(delta)) - (Y.ominus(X) + (J_ominus_X @ delta))
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 6)


def test_jacobian_Y_ominus_X_wrt_Y():
    X = SE3((SO3.from_roll_pitch_yaw(np.pi / 8, np.pi / 2, 5 * np.pi / 6), np.array([[2, 1, 1]]).T))
    Y = SE3((SO3.from_roll_pitch_yaw(np.pi / 7, np.pi / 3, 4 * np.pi / 6), np.array([[2, 1, 0]]).T))

    J_ominus_Y = Y.jac_Y_ominus_X_wrt_Y(X)

    # Should be J_r_inv.
    np.testing.assert_equal(J_ominus_Y, SE3.jac_right_inverse(Y - X))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((6, 1))
    taylor_diff = Y.oplus(delta).ominus(X) - (Y.ominus(X) + (J_ominus_Y @ delta))
    np.testing.assert_almost_equal(taylor_diff, np.zeros((6, 1)), 6)


def test_has_len_that_returns_correct_dimension():
    X = SE3()

    # Should have 6 DOF.
    np.testing.assert_equal(len(X), 6)
