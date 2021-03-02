import numpy as np
from pylie import SO3


class SE3:
    """Represents an element of the SE(3) Lie group (poses in 3D)."""

    def __init__(self, pose_tuple=(SO3(), np.zeros((3, 1)))):
        """Constructs an SE(3) element.
        The default is the identity element.

        :param pose_tuple: A tuple (rotation (SO3), translation (3D column vector) (optional).
        """
        self.rotation, self.translation = pose_tuple

    @classmethod
    def from_matrix(cls, T):
        """Construct an SE(3) element corresponding from a pose matrix.
        The rotation is fitted to the closest rotation matrix, the bottom row of the 4x4 matrix is ignored.

        :param T: 4x4 or 3x4 pose matrix.
        :return: The SE(3) element.
        """

        return cls((SO3(T[:3, :3]), T[:3, 3:4]))

    @property
    def rotation(self):
        """ The so3 rotation, an element of SO(3)

        :return: An SO3 object corresponding to the orientation.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, so3):
        """Sets the rotation

        :param so3: An SO3
        """
        if not isinstance(so3, SO3):
            raise TypeError('Rotation must be a SO3')

        self._rotation = so3

    @property
    def translation(self):
        """The translation, a 3D column vector

        :return: A 3D column vector corresponding to the translation.
        """
        return self._translation

    @translation.setter
    def translation(self, t):
        """Sets the translation

        :param t: 3D column vector
        """
        if not isinstance(t, np.ndarray) and t.shape == (3, 1):
            raise TypeError('Translation must be a 3D column vector')

        self._translation = t

    def to_matrix(self):
        """Return the matrix representation of this pose.

        :return: 4x4 SE(3) matrix
        """
        T = np.identity(4)
        T[0:3, 0:3] = self.rotation.matrix
        T[0:3, 3] = self.translation.T
        return T

    def to_tuple(self):
        """Return the tuple representation of this pose

        :return: (R (3x3 matrix), t (3D column vector)
        """
        return (self.rotation.matrix, self.translation)

    def compose(X, Y):
        """Compose this element X with another element Y on the right

        :param Y: The other Pose3 element
        :return: This element X composed with Y
        """
        return SE3((X.rotation @ Y.rotation, X.rotation * Y.translation + X.translation))

    def inverse(self):
        """Compute the inverse of the current element X.

        :return: The inverse of the current element.
        """
        rot_inv = self.rotation.inverse()
        return SE3((rot_inv, -(rot_inv * self.translation)))

    def action(self, x):
        """Perform the action of the SE(3) element on the 3D column vector x.

        :param x: 3D column vector to be transformed (or a matrix of 3D column vectors)
        :return: The resulting rotated and translated 3D column vectors
        """
        return self.rotation * x + self.translation

    def adjoint(self):
        """The adjoint at the element.
        :return: The adjoint, a 6x6 matrix.
        """
        R = self.rotation.matrix

        return np.block([[R, SO3.hat(self.translation) @ R],
                         [np.zeros((3, 3)), R]])

    def oplus(X, xi_vec):
        """Computes the right perturbation of Exp(xi_vec) on the element X.

        :param xi_vec: The tangent space vector, a 6D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The perturbed SE3 element Y = X :math:`\\oplus` xi_vec.
        """
        if not (isinstance(xi_vec, np.ndarray) and xi_vec.shape == (6, 1)):
            raise TypeError('Argument must be a 6D column vector')

        return X @ SE3.Exp(xi_vec)

    def ominus(Y, X):
        """Computes the tangent space vector at X between X and this element Y.

        :param X: The other element
        :return: The difference xi_vec = Y :math:'\\ominus' X
        """
        if not isinstance(X, SE3):
            raise TypeError('Argument must be an SE3')
        return (X.inverse() @ Y).Log()

    def Log(self):
        """Computes the tangent space vector xi_vec at the current element X.

        :return: The tangent space vector xi_vec = [rho_vec, theta_vec]^T.
        """
        theta, u_vec = self.rotation.Log(split_angle_axis=True)

        if theta == 0:
            return np.vstack((self.translation, np.zeros((3, 1))))

        theta_vec = theta * u_vec

        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / (theta ** 2)

        theta_hat = SO3.hat(theta_vec)
        V_inv = np.identity(3) - 0.5 * theta_hat + np.linalg.matrix_power(theta_hat, 2) * (
                1 - a / (2 * b)) / (theta ** 2)

        rho_vec = V_inv @ self.translation

        return np.vstack((rho_vec, theta_vec))

    def jac_inverse_X_wrt_X(X):
        """Computes the Jacobian of the inverse operation X.inverse() with respect to the element X.

        :return: The Jacobian (6x6 matrix)
        """
        return -X.adjoint()

    def jac_action_Xx_wrt_X(X, x):
        """Computes the Jacobian of the action X.action(x) with respect to the element X.

        :param x: The 3D column vector x.
        :return: The Jacobian (6x3 matrix)
        """
        return np.block([[X.rotation.matrix, -(X.rotation.matrix @ SO3.hat(x))]])

    def jac_action_Xx_wrt_x(X):
        """Computes the Jacobian of the action X.action(x) with respect to the element X.

        :return: The Jacobian (3x3 matrix)
        """
        return X.rotation.matrix

    def jac_Y_ominus_X_wrt_X(Y, X):
        """Compute the Jacobian of Y.ominus(X) with respect to the element X.

        :param X: The SE(3) element X.
        :return: The Jacobian (6x6 matrix)
        """
        return -SE3.jac_left_inverse(Y - X)

    def jac_Y_ominus_X_wrt_Y(Y, X):
        """Compute the Jacobian of Y.ominus(X) with respect to the element Y.

        :param X: The SE(3) element X.
        :return: The Jacobian (6x6 matrix)
        """
        return SE3.jac_right_inverse(Y - X)

    def __mul__(self, other):
        """Multiplication operator performs action on vectors.

        :param other: 3D column vector, or a matrix of 3D column vectors.
        :return: Transformed 3D column vector
        """
        if isinstance(other, np.ndarray) and other.shape[0] == 3:
            # Other is matrix of 3D column vectors, perform action on vectors.
            return self.action(other)
        else:
            raise TypeError('Argument must be a matrix of 3D column vectors')

    def __matmul__(self, other):
        """Matrix multiplication operator performs composition on elements of SE(3).

        :param other: Other SE3
        :return: Composed SE3
        """
        if isinstance(other, SE3):
            # Other is SE3, perform composition.
            return self.compose(other)
        else:
            raise TypeError('Argument must be an SE3')

    def __add__(self, xi_vec):
        """Add operator performs the "oplus" operation on the element X.

        :param xi_vec: The tangent space vector, a 6D column vector xi_vec = [rho_vec, theta_vec]^T..
        :return: The perturbed SE3 element Y = X :math:`\\oplus` xi_vec.
        """
        return self.oplus(xi_vec)

    def __sub__(self, X):
        """Subtract operator performs the "ominus" operation at X between X and this element Y.

        :param X: The other element
        :return: The difference xi_vec = Y :math:'\\ominus' X
        """
        return self.ominus(X)

    def __len__(self):
        """Length operator returns the dimension of the tangent vector space,
        which is equal to the number of degrees of freedom (DOF).

        :return: The DOF for poses (6)
        """
        return 6

    def __repr__(self):
        """Formal string representation of the object.

        :return: The formal representation as a string
        """
        return "SE3({\n" + repr(self.rotation) + ",\n" + repr(self.translation) + "\n})"

    def __str__(self):
        """Informal string representation of the object
        prints the matrix representation.

        :return: The matrix representation as a string
        """
        return str(self.to_matrix())

    @staticmethod
    def hat(xi_vec):
        """Performs the hat operator on the tangent space vector xi_vec,
        which returns the corresponding Lie Algebra matrix xi_hat.

        :param xi_vec: 6d tangent space column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Lie Algebra (4x4 matrix).
        """
        return np.block([[SO3.hat(xi_vec[3:]), xi_vec[:3]],
                         [np.zeros((1, 4))]])

    @staticmethod
    def vee(xi_hat):
        """Performs the vee operator on the Lie Algebra matrix xi_hat,
        which returns the corresponding tangent space vector.

        :param xi_hat: The Lie Algebra (4x4 matrix)
        :return: 6d tangent space column vector xi_vec = [rho_vec, theta_vec]^T.
        """
        return np.vstack((xi_hat[:3, 3:4], SO3.vee(xi_hat[:3, :3])))

    @staticmethod
    def Exp(xi_vec):
        """Computes the Exp-map on the Lie algebra vector xi_vec,
        which transfers it to the corresponding Lie group element.

        :param xi_vec: 6d tangent space column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: Corresponding SE(3) element
        """
        xi_hat = SE3.hat(xi_vec)
        theta = np.linalg.norm(xi_vec[3:])

        if theta < 1e-10:
            return SE3.from_matrix(np.identity(4) + xi_hat)
        else:
            return SE3.from_matrix(
                np.identity(4) + xi_hat + ((1 - np.cos(theta)) / (theta ** 2)) * np.linalg.matrix_power(xi_hat, 2) +
                ((theta - np.sin(theta)) / (theta ** 3)) * np.linalg.matrix_power(xi_hat, 3))

    @staticmethod
    def jac_composition_XY_wrt_X(Y):
        """Computes the Jacobian of the composition X.compose(Y) with respect to the element X.

        :param Y: SE3 element Y
        :return: The Jacobian (6x6 matrix)
        """
        R_Y_inv = Y.rotation.inverse().matrix
        return np.block([[R_Y_inv, -(R_Y_inv @ SO3.hat(Y.translation))],
                         [np.zeros((3, 3)), R_Y_inv]])

    @staticmethod
    def jac_composition_XY_wrt_Y():
        """Computes the Jacobian of the composition X.compose(Y) with respect to the element Y.

        :return: The Jacobian (6x6 identity matrix)
        """
        return np.identity(6)

    @staticmethod
    def _Q_left(xi_vec):
        rho_vec = xi_vec[:3]
        theta_vec = xi_vec[3:]
        theta = np.linalg.norm(theta_vec)

        if theta < 1e-10:
            return np.zeros((3, 3))

        rho_hat = SO3.hat(rho_vec)
        theta_hat = SO3.hat(theta_vec)

        return 0.5 * rho_hat + ((theta - np.sin(theta)) / theta ** 3) * \
               (theta_hat @ rho_hat + rho_hat @ theta_hat + theta_hat @ rho_hat @ theta_hat) - \
               ((1 - 0.5 * theta ** 2 - np.cos(theta)) / theta ** 4) * \
               (theta_hat @ theta_hat @ rho_hat + rho_hat @ theta_hat @ theta_hat -
                3 * theta_hat @ rho_hat @ theta_hat) - \
               0.5 * ((1 - 0.5 * theta ** 2 - np.cos(theta)) / theta ** 4 - 3 *
                      ((theta - np.sin(theta) - (theta ** 3 / 6)) / theta ** 5)) * \
               (theta_hat @ rho_hat @ theta_hat @ theta_hat + theta_hat @ theta_hat @ rho_hat @ theta_hat)

    @staticmethod
    def _Q_right(xi_vec):
        return SE3._Q_left(-xi_vec)

    @staticmethod
    def jac_right(xi_vec):
        """Compute the right derivative of Exp(xi_vec) with respect to xi_vec.

        :param xi_vec: The tangent space 6D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Jacobian (6x6 matrix)
        """
        theta_vec = xi_vec[3:]

        J_r_theta = SO3.jac_right(theta_vec)
        Q_r = SE3._Q_right(xi_vec)

        return np.block([[J_r_theta, Q_r],
                         [np.zeros((3, 3)), J_r_theta]])

    @staticmethod
    def jac_left(xi_vec):
        """Compute the left derivative of Exp(xi_vec) with respect to xi_vec.

        :param xi_vec: The tangent space 6D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Jacobian (6x6 matrix)
        """
        theta_vec = xi_vec[3:]

        J_l_theta = SO3.jac_left(theta_vec)
        Q_l = SE3._Q_left(xi_vec)

        return np.block([[J_l_theta, Q_l],
                         [np.zeros((3, 3)), J_l_theta]])

    @staticmethod
    def jac_right_inverse(xi_vec):
        """Compute the right derivative of Log(X) with respect to X for xi_vec = Log(X).

        :param xi_vec: The tangent space 6D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Jacobian (6x6 matrix)
        """
        theta_vec = xi_vec[3:]

        J_r_inv_theta = SO3.jac_right_inverse(theta_vec)
        Q_r = SE3._Q_right(xi_vec)

        return np.block([[J_r_inv_theta, -J_r_inv_theta @ Q_r @ J_r_inv_theta],
                         [np.zeros((3, 3)), J_r_inv_theta]])

    @staticmethod
    def jac_left_inverse(xi_vec):
        """Compute the left derivative of Log(X) with respect to X for xi_vec = Log(X).

        :param xi_vec: The tangent space 6D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Jacobian (6x6 matrix)
        """
        theta_vec = xi_vec[3:]

        J_l_inv_theta = SO3.jac_left_inverse(theta_vec)
        Q_l = SE3._Q_left(xi_vec)

        return np.block([[J_l_inv_theta, -J_l_inv_theta @ Q_l @ J_l_inv_theta],
                         [np.zeros((3, 3)), J_l_inv_theta]])

    @staticmethod
    def jac_X_oplus_tau_wrt_X(xi_vec):
        """Compute the Jacobian of X.oplus(tau) with respect to the element X

        :param xi_vec: The tangent space 6D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Jacobian (6x6 matrix)
        """
        return SE3.Exp(xi_vec).inverse().adjoint()

    @staticmethod
    def jac_X_oplus_tau_wrt_tau(xi_vec):
        """Compute the Jacobian of X.oplus(tau) with respect to the tangent space vector tau

        :param xi_vec: The tangent space 6D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Jacobian (6x6 matrix)
        """
        return SE3.jac_right(xi_vec)
