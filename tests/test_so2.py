import numpy as np
from pylie import SO2
import pytest

def test_construct_identity_with_no_args():
    so2 = SO2()
    np.testing.assert_equal(so2.angle(), 0.0)
    np.testing.assert_array_equal(so2.to_matrix(), np.identity(2))


def test_construct_with_real_and_imag_args():
    angle = np.pi/10
    real = np.cos(angle)
    imag = np.sin(angle)
    so2 = SO2(real, imag)
    np.testing.assert_almost_equal(so2.real(), real, 16)
    np.testing.assert_almost_equal(so2.imag(), imag, 16)
    np.testing.assert_almost_equal(so2.angle(), angle, 16)


def test_construct_with_unnormalised_real_and_imag_args():
    angle = np.pi/5
    real = np.cos(angle) * 10
    imag = np.sin(angle) * 10
    so2 = SO2(real, imag)
    np.testing.assert_equal(so2.real(), real/10)
    np.testing.assert_equal(so2.imag(), imag/10)
    np.testing.assert_equal(so2.angle(), angle)


def test_construct_with_zero_norm_fails():
    with pytest.raises(ValueError):
        SO2(0, 0)

    with pytest.raises(ValueError):
        SO2(1e-20, 1e-20)


def test_construct_with_one_argument_fails():
    with pytest.raises(ValueError):
        so2 = SO2(1)

    with pytest.raises(ValueError):
        so2 = SO2(imag=1)

    with pytest.raises(ValueError):
        so2 = SO2(real=1)

def test_construct_from_angle():
    for i in np.arange(-12, 12):
        theta = i/12 * 2 * np.pi
        so2 = SO2.from_angle(theta)
        np.testing.assert_almost_equal(so2.real(), np.cos(theta), 16)
        np.testing.assert_almost_equal(so2.imag(), np.sin(theta), 16)

def test_construct_from_matrix():
    theta = np.pi/3
    R = SO2.from_angle(theta).to_matrix()
    so2 = SO2.from_matrix(R)
    np.testing.assert_equal(so2.angle(), theta)
    np.testing.assert_array_almost_equal(so2.to_matrix(), R, 14)

    # Matrices not elements of SO(2) should be fitted to SO(2)
    R_noisy = R + 0.1 + np.random.rand(2)
    so2_fitted = SO2.from_matrix(R_noisy)
    R_fitted = so2_fitted.to_matrix()

    assert np.any(np.not_equal(R_fitted, R_noisy))
    np.testing.assert_almost_equal(np.linalg.det(R_fitted), 1, 14)
    np.testing.assert_almost_equal((R_fitted.T @ R_fitted), np.identity(2), 14)


def test_hat_returns_skew_symmetric_matrix():
    theta = 1.0
    theta_hat = SO2.hat(theta)
    assert theta_hat[0, 0] == 0
    assert theta_hat[0, 1] == -theta
    assert theta_hat[1, 0] == theta
    assert theta_hat[1, 1] == 0


def test_vee_extracts_correct_angle_from_skew_symmetric_matrix():
    theta = 3.0
    theta_hat = SO2.hat(theta)
    theta_hat_vee = SO2.vee(theta_hat)
    np.testing.assert_array_equal(theta_hat_vee, theta)


def test_exp_returns_correct_rotation():
    for i in np.arange(0, 12):
        theta = i/12 * 2 * np.pi
        so2 = SO2.Exp(theta)
        np.testing.assert_almost_equal(so2.angle(), np.arctan2(np.sin(theta), np.cos(theta)), 14)

    theta = 2 * np.pi
    so2 = SO2.Exp(theta)
    np.testing.assert_almost_equal(so2.angle(), 0, 14)


def test_log_returns_correct_theta():
    for i in np.arange(0, 12):
        theta = i/12 * 2 * np.pi
        so2 = SO2.Exp(theta)
        theta_log = so2.Log()

        np.testing.assert_almost_equal(theta_log, so2.angle(), 16)


def test_inverse_returns_transposed():
    so2 = SO2.from_angle(np.pi / 4)
    so2_inv = so2.inverse()

    np.testing.assert_almost_equal(so2_inv.to_matrix(), so2.to_matrix().T, 14)
    np.testing.assert_almost_equal(so2_inv.inverse().to_matrix(), so2.to_matrix(), 14)


def test_action_on_vector_works():
    unit_x = np.array([[1, 0]]).T
    unit_y = np.array([[0, 1]]).T
    so2 = SO2.from_angle(np.pi / 2)

    np.testing.assert_almost_equal(so2.action(unit_x), unit_y, 14)


def test_action_on_vector_with_operator_works():
    unit_x = np.array([[1, 0]]).T
    so2 = SO2.from_angle(np.pi)

    np.testing.assert_almost_equal(so2 * unit_x, -unit_x, 14)


def test_composition_with_identity_works():
    so2 = SO2.from_angle(np.pi / 4)

    comp_with_identity = so2.compose(SO2())
    comp_from_identity = SO2().compose(so2)

    np.testing.assert_almost_equal(comp_with_identity.to_matrix(), so2.to_matrix(), 14)
    np.testing.assert_almost_equal(comp_from_identity.to_matrix(), so2.to_matrix(), 14)


def test_composition_returns_correct_rotation():
    so2_180 = SO2.from_angle(np.pi)
    so2_90 = SO2.from_angle(np.pi / 2)

    so2_comp = so2_180.compose(so2_90)

    expected = np.array([[ 0, 1],
                         [-1, 0]])

    np.testing.assert_almost_equal(so2_comp.to_matrix(), expected, 14)


def test_composition_with_operator_works():
    so2_45 = SO2.from_angle(np.pi / 4)

    so2_comp = so2_45 @ so2_45

    expected = np.array([[0, -1],
                         [1, 0]])

    np.testing.assert_almost_equal(so2_comp.to_matrix(), expected, 14)


def test_adjoint_returns_rotation_one():
    so2_1 = SO2()
    so2_2 = SO2.from_angle(3 * np.pi / 4)

    np.testing.assert_array_equal(so2_1.adjoint(), 1.0)
    np.testing.assert_array_equal(so2_2.adjoint(), 1.0)


def test_perturbation_by_adding_a_simple_rotation_works():
    ident = SO2()
    pert = ident.oplus(np.pi / 2)
    expected = np.array([[0, -1],
                         [1, 0]])

    np.testing.assert_almost_equal(pert.to_matrix(), expected, 14)


def test_perturbation_by_adding_a_simple_rotation_with_operator_works():
    ident = SO2()
    pert = ident + (-np.pi)
    expected = np.array([[-1, 0],
                         [0, -1]])

    np.testing.assert_almost_equal(pert.to_matrix(), expected, 14)


def test_difference_for_simple_rotation_works():
    X = SO2()
    theta = np.pi / 3
    Y = SO2.Exp(theta)
    theta_diff = Y.ominus(X)

    np.testing.assert_array_equal(theta_diff, theta)


def test_difference_for_simple_rotation_with_operator_works():
    X = SO2()
    theta = 3 * np.pi / 4
    Y = SO2.Exp(theta)
    theta_diff = Y - X

    np.testing.assert_almost_equal(theta_diff, theta, 14)


def test_jacobian_inverse():
    theta = np.pi / 3
    X = SO2.from_angle(theta)

    J_inv = X.jac_inverse_X_wrt_X()

    # Jacobian should be -X.adjoint().
    np.testing.assert_array_equal(J_inv, -X.adjoint())

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = X.oplus(delta).inverse() - X.inverse().oplus(J_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, 0.0, 14)


def test_jacobian_composition_XY_wrt_X():
    theta_x = np.pi / 4
    theta_y = np.pi / 2
    X = SO2.from_angle(theta_x)
    Y = SO2.from_angle(theta_y)

    J_comp_X = X.jac_composition_XY_wrt_X(Y)

    # Jacobian should be Y.inverse().adjoint()
    np.testing.assert_array_equal(J_comp_X, Y.inverse().adjoint())

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = X.oplus(delta).compose(Y) - X.compose(Y).oplus(J_comp_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, 0, 14)


def test_jacobian_composition_XY_wrt_Y():
    theta_x = np.pi / 4
    theta_y = -np.pi / 3
    X = SO2.from_angle(theta_x)
    Y = SO2.from_angle(theta_y)

    J_comp_Y = X.jac_composition_XY_wrt_Y()

    # Jacobian should be identity
    np.testing.assert_array_equal(J_comp_Y, 1.0)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = X.compose(Y.oplus(delta)) - X.compose(Y).oplus(J_comp_Y @ delta)
    np.testing.assert_almost_equal(taylor_diff, 0.0, 14)


def test_jacobian_action_Xx_wrt_X():
    theta_x = 3 * np.pi / 4
    X = SO2.from_angle(theta_x)
    x = np.array([[1, 2]]).T

    J_action_X = X.jac_action_Xx_wrt_X(x)

    # Jacobian should be -X.matrix * SO3.hat(x).
    np.testing.assert_array_equal(J_action_X, X.to_matrix() @ SO2.hat(1.0) @ x)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = X.oplus(delta).action(x) - (X.action(x) + J_action_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, 0.0, 5)


def test_jacobian_action_Xx_wrt_x():
    theta_x = 3 * np.pi / 4
    X = SO2.from_angle(theta_x)
    x = np.array([[1, 2]]).T

    J_action_x = X.jac_action_Xx_wrt_x()

    # Jacobian should be X.to_matrix().
    np.testing.assert_array_equal(J_action_x, X.to_matrix())

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((2, 1))
    taylor_diff = X.action(x + delta) - (X.action(x) + J_action_x @ delta)
    np.testing.assert_almost_equal(taylor_diff, np.zeros((2, 1)), 14)


def test_jacobian_right():
    theta = 3 * np.pi / 4

    J_r = SO2.jac_right(theta)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = SO2.Exp(theta + delta) - (SO2.Exp(theta) + J_r @ delta)
    np.testing.assert_almost_equal(taylor_diff, 0.0, 5)


def test_jacobian_left():
    theta = np.pi / 5

    J_l = SO2.jac_left(theta)

    # Should have J_l(theta) == J_r(-theta).
    np.testing.assert_almost_equal(J_l, SO2.jac_right(-theta), 14)


def test_jacobian_right_inverse():
    theta = 3 * np.pi / 4
    X = SO2.Exp(theta)

    J_r_inv = SO2.jac_right_inverse(theta)

    # Should have J_l * J_r_inv = Exp(theta).adjoint().
    J_l = SO2.jac_left(theta)
    np.testing.assert_almost_equal(J_l @ J_r_inv, SO2.Exp(theta).adjoint(), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = X.oplus(delta).Log() - (X.Log() + J_r_inv @ delta)
    np.testing.assert_almost_equal(taylor_diff, 0.0, 5)


def test_jacobian_left_inverse():
    theta = np.pi / 4

    # Should have J_l_inv(theta) == J_r_inv(theta).T
    np.testing.assert_almost_equal(SO2.jac_left_inverse(theta), SO2.jac_right_inverse(theta).T, 14)


def test_jacobian_X_oplus_tau_wrt_X():
    theta_x = 3 * np.pi / 4
    X = SO2.from_angle(theta_x)

    theta = np.pi / 4
    J_oplus_X = X.jac_X_oplus_tau_wrt_X(theta)

    # Should be Exp(tau).adjoint().inverse()
    np.testing.assert_almost_equal(J_oplus_X, SO2.Exp(theta).adjoint(), 14)

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = X.oplus(delta).oplus(theta) - X.oplus(theta).oplus(J_oplus_X @ delta)
    np.testing.assert_almost_equal(taylor_diff, 0.0, 14)


def test_jacobian_X_oplus_tau_wrt_tau():
    theta_x = np.pi / 4
    X = SO2.from_angle(theta_x)

    theta_vec = np.pi / 3
    J_oplus_tau = X.jac_X_oplus_tau_wrt_tau(theta_vec)

    # Should be J_r.
    np.testing.assert_equal(J_oplus_tau, X.jac_right(theta_vec))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = X.oplus(theta_vec + delta) - X.oplus(theta_vec).oplus(J_oplus_tau @ delta)
    np.testing.assert_almost_equal(taylor_diff, 0.0, 6)


def test_jacobian_Y_ominus_X_wrt_X():
    theta_x = np.pi / 4
    theta_y = np.pi / 2
    X = SO2.from_angle(theta_x)
    Y = SO2.from_angle(theta_y)

    J_ominus_X = Y.jac_Y_ominus_X_wrt_X(X)

    # Should be -J_l_inv.
    np.testing.assert_equal(J_ominus_X, -SO2.jac_left_inverse(Y - X))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = Y.ominus(X.oplus(delta)) - (Y.ominus(X) + (J_ominus_X @ delta))
    np.testing.assert_almost_equal(taylor_diff, 0.0, 6)


def test_jacobian_Y_ominus_X_wrt_Y():
    theta_x = np.pi / 4
    theta_y = np.pi / 3
    X = SO2.from_angle(theta_x)
    Y = SO2.from_angle(theta_y)

    J_ominus_Y = Y.jac_Y_ominus_X_wrt_Y(X)

    # Should be J_r_inv.
    np.testing.assert_equal(J_ominus_Y, SO2.jac_right_inverse(Y - X))

    # Test the Jacobian numerically.
    delta = 1e-3 * np.ones((1, 1))
    taylor_diff = Y.oplus(delta).ominus(X) - (Y.ominus(X) + (J_ominus_Y @ delta))
    np.testing.assert_almost_equal(taylor_diff, 0.0, 6)


def test_has_len_that_returns_correct_dimension():
    X = SO2()

    # Should have 1 DOF.
    np.testing.assert_equal(len(X), 1)


def test_string_representation_is_matrix():
    X = SO2()

    # Should be the same as the string representation of the matrix.
    np.testing.assert_equal(str(X), str(X.to_matrix()))
