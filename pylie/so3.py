import numpy as np


class SO3:
    """Represents an element of the SO(3) Lie group (rotations in 3D)."""

    def __init__(self, R=np.identity(3)):
        """Constructs an SO(3) element.
        The default is the identity element.
        Other 3x3 matrices R are fitted to the closest matrix on SO(3).

        :param R: A 3x3 rotation matrix (optional).
        """
        if R is self.__init__.__defaults__[0]:
            # Default argument is identity.
            # Set property directly, since guaranteed SO(3).
            self._matrix = np.identity(3)
        else:
            # Argument should be some 3x3 matrix.
            # Fit to SO(3).
            self.matrix = R

    @classmethod
    def from_angle_axis(cls, angle, axis):
        """Construct an SO(3) element corresponding to a rotation around a specified axis.

        :param angle: Rotation angle in radians.
        :param axis: Rotation axis as a unit 3d column vector.
        :return: The SO(3) element.
        """
        # Guaranteed to be SO(3), so set property directly.
        so3 = cls()
        so3._matrix = cls.Exp(angle * axis).matrix
        return so3

    @classmethod
    def rot_x(cls, angle):
        """Construct an SO(3) element corresponding to a rotation around the x-axis.

        :param angle: Rotation angle in radians.
        :return: The SO(3) element.
        """
        return cls.from_angle_axis(angle, np.array([[1, 0, 0]]).T)

    @classmethod
    def rot_y(cls, angle):
        """Construct an SO(3) element corresponding to a rotation around the y-axis.

        :param angle: Rotation angle in radians.
        :return: The SO(3) element.
        """
        return cls.from_angle_axis(angle, np.array([[0, 1, 0]]).T)

    @classmethod
    def rot_z(cls, angle):
        """Construct an SO(3) element corresponding to a rotation around the z-axis.

        :param angle: Rotation angle in radians.
        :return: The SO(3) element.
        """
        return cls.from_angle_axis(angle, np.array([[0, 0, 1]]).T)

    @classmethod
    def from_roll_pitch_yaw(cls, roll, pitch, yaw):
        """Construct an SO(3) element from Z-Y-X Euler angles.

        :param roll: Rotation angle around the x-axis in radians.
        :param pitch: Rotation angle around the y-axis in radians.
        :param yaw: Rotation angle around the z-axis in radians.
        :return: The SO(3) element.
        """
        # Guaranteed to be SO(3) through composition, so set property directly.
        so3 = cls()
        so3._matrix = (SO3.rot_z(yaw) @ SO3.rot_y(pitch) @ SO3.rot_x(roll)).matrix
        return so3

    @property
    def matrix(self):
        """ The matrix representation of the SO(3) element

        :return: 3x3 rotation matrix corresponding to this SO(3) element.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, R):
        """Sets the matrix to the closest element on SO(3)

        :param R: 3x3 matrix
        """
        # This is slower than necessary, but ensures correct representation.
        self._matrix = SO3.to_so3_matrix(R)

    @staticmethod
    def to_so3_matrix(R):
        """Fits an arbitrary 3x3 matrix to the closest element on SO(3)

        :param R: An arbitrary 3x3 matrix
        :return: The closest valid 3x3 rotation matrix
        """
        if not (isinstance(R, np.ndarray) and R.shape == (3, 3)):
            raise TypeError('Argument must be a 3x3 matrix')

        u, s, v = np.linalg.svd(R)
        R = u.dot(v)

        if np.linalg.det(R) < 0:
            R = -R

        return R

    def Log(self, split_angle_axis=False):
        """Computes the tangent space vector at the current element X.

        :param split_angle_axis: Split tangent space vector into angle-axis fields? (optional)
        :return: The tangent space vector theta_vec, or angle, axis if split_angle_axis is True.
        """
        R = self.matrix

        theta = np.arccos(np.clip(0.5 * (R.trace() - 1), -1, 1))

        if theta < 1e-10:
            theta = 0
            u_vec = np.array([[0, 0, 0]]).T
        else:
            u_vec = SO3.vee(R - R.T) / (2 * np.sin(theta))

        if split_angle_axis:
            return theta, u_vec
        else:
            return theta * u_vec

    def inverse(self):
        """Compute the inverse of the current element X.

        :return: The inverse of the current element.
        """
        # The transpose is guaranteed to be SO(3), update the property directly.
        X_inv = SO3()
        X_inv._matrix = self.matrix.T
        return X_inv

    def action(self, x):
        """Perform the action of the SO(3) element on the 3D column vector x.

        :param x: 3D column vector to be transformed (or a matrix of 3D column vectors)
        :return: The resulting rotated 3D column vectors
        """
        return self.matrix @ x

    def compose(self, Y):
        """Compose this element with another element on the right

        :param Y: The other SO3 element
        :return: This element composed with Y
        """
        return SO3(self.matrix @ Y.matrix)

    def adjoint(self):
        """The adjoint at the element.
        :return: The adjoint, a 3x3 rotation matrix.
        """
        return self.matrix

    def oplus(self, theta_vec):
        """Computes the right perturbation of Exp(theta_vec) on the element X.

        :param theta_vec: The tangent space vector, a 3D column vector.
        :return: The perturbed SO3 element Y = X :math:`\\oplus` theta_vec.
        """
        if not (isinstance(theta_vec, np.ndarray) and theta_vec.shape == (3, 1)):
            raise TypeError('Argument must be a 3D column vector')

        return self @ SO3.Exp(theta_vec)

    def ominus(self, X):
        """Computes the tangent space vector at X between X and this element Y.

        :param X: The other element
        :return: The difference theta_vec = Y :math:'\\ominus' X
        """
        if not isinstance(X, SO3):
            raise TypeError('Argument must be an SO3')
        return (X.inverse() @ self).Log()

    def jac_inverse_X_wrt_X(X):
        """Computes the Jacobian of the inverse operation X.inverse() with respect to the element X.

        :return: The Jacobian (3x3 matrix)
        """
        return -X.matrix

    def jac_action_Xx_wrt_X(X, x):
        """Computes the Jacobian of the action X.action(x) with respect to the element X.

        :param x: The 3D column vector x.
        :return: The Jacobian (3x3 matrix)
        """
        return -X.matrix @ SO3.hat(x)

    def jac_action_Xx_wrt_x(X):
        """Computes the Jacobian of the action X.action(x) with respect to the element X.

        :return: The Jacobian (3x3 matrix)
        """
        return X.matrix

    def jac_Y_ominus_X_wrt_X(Y, X):
        """Compute the Jacobian of Y.ominus(X) with respect to the element X.

        :param X: The SO(3) element X.
        :return: The Jacobian (3x3 matrix)
        """
        return -SO3.jac_left_inverse(Y - X)

    def jac_Y_ominus_X_wrt_Y(Y, X):
        """Compute the Jacobian of Y.ominus(X) with respect to the element Y.

        :param X: The SO(3) element X.
        :return: The Jacobian (3x3 matrix)
        """
        return SO3.jac_right_inverse(Y - X)

    def __add__(self, theta_vec):
        """Add operator performs the "oplus" operation on the element X.

        :param theta_vec: The tangent space vector, a 3D column vector.
        :return: The perturbed SO3 element Y = X :math:`\\oplus` theta_vec.
        """
        return self.oplus(theta_vec)

    def __sub__(self, X):
        """Subtract operator performs the "ominus" operation at X between X and this element Y.

        :param X: The other element
        :return: The difference theta_vec = Y :math:'\\ominus' X
        """
        return self.ominus(X)

    def __mul__(self, other):
        """Multiplication operator performs action on vectors.

        :param other: 3D column vector, or a matrix of 3D column vectors
        :return: Transformed 3D column vectors
        """
        if isinstance(other, np.ndarray) and other.shape[0] == 3:
            # Other is matrix of 3D column vectors, perform action on vectors.
            return self.action(other)
        else:
            raise TypeError('Argument must be a matrix of 3D column vectors')

    def __matmul__(self, other):
        """Matrix multiplication operator performs composition on elements of SO(3).

        :param other: Other SO3
        :return: Composed SO3
        """
        if isinstance(other, SO3):
            # Other is SO3, perform composition.
            return self.compose(other)
        else:
            raise TypeError('Argument must be an SO3')

    @staticmethod
    def hat(theta_vec):
        """Performs the hat operator on the tangent space vector theta_vec,
        which returns the corresponding skew symmetric Lie Algebra matrix theta_hat.

        :param theta_vec: 3d tangent space column vector.
        :return: The Lie Algebra (3x3 matrix).
        """
        theta_vec = theta_vec.flatten()
        return np.array([[0, -theta_vec[2], theta_vec[1]]
                            , [theta_vec[2], 0, -theta_vec[0]]
                            , [-theta_vec[1], theta_vec[0], 0]])

    @staticmethod
    def vee(theta_hat):
        """Performs the vee operator on the skew symmetric Lie Algebra matrix theta_hat,
        which returns the corresponding tangent space vector.

        :param theta_hat: The Lie Algebra (3x3 matrix)
        :return: 3d tangent space column vector.
        """
        return np.array([[theta_hat[2, 1], theta_hat[0, 2], theta_hat[1, 0]]]).T

    @staticmethod
    def Exp(theta_vec):
        """Computes the Exp-map on the Lie algebra vector theta_vec,
        which transfers it to the corresponding Lie group element.

        :param theta_vec: 3d tangent space column vector.
        :return: Corresponding SO(3) element
        """
        theta = np.linalg.norm(theta_vec)

        if theta < 1e-10:
            return SO3()

        u_vec = theta_vec / theta
        u_hat = SO3.hat(u_vec)

        R = np.identity(3) + np.sin(theta) * u_hat + (1 - np.cos(theta)) * (u_hat @ u_hat)

        return SO3(R)

    @staticmethod
    def jac_composition_XY_wrt_X(Y):
        """Computes the Jacobian of the composition X.compose(Y) with respect to the element X.

        :param Y: SO3 element Y
        :return: The Jacobian (3x3 matrix)
        """
        return Y.matrix.T

    @staticmethod
    def jac_composition_XY_wrt_Y():
        """Computes the Jacobian of the composition X.compose(Y) with respect to the element Y.

        :return: The Jacobian (3x3 matrix)
        """
        return np.identity(3)

    @staticmethod
    def jac_right(theta_vec):
        """Compute the right derivative of Exp(theta_vec) with respect to theta_vec.

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        theta = np.linalg.norm(theta_vec)
        if theta < 1e-10:
            return np.identity(3)

        theta_hat = SO3.hat(theta_vec)

        return np.identity(3) - ((1 - np.cos(theta)) / (theta ** 2)) * theta_hat + (
                (theta - np.sin(theta)) / (theta ** 3)) * theta_hat @ theta_hat

    @staticmethod
    def jac_left(theta_vec):
        """Compute the left derivative of Exp(theta_vec) with respect to theta_vec.

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        theta = np.linalg.norm(theta_vec)
        if theta < 1e-10:
            return np.identity(3)

        theta_hat = SO3.hat(theta_vec)

        return np.identity(3) + ((1 - np.cos(theta)) / (theta ** 2)) * theta_hat + (
                (theta - np.sin(theta)) / (theta ** 3)) * theta_hat @ theta_hat

    @staticmethod
    def jac_right_inverse(theta_vec):
        """Compute the right derivative of Log(X) with respect to X for theta_vec = Log(X).

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        theta = np.linalg.norm(theta_vec)
        if theta < 1e-10:
            return np.identity(3)

        theta_hat = SO3.hat(theta_vec)

        return np.identity(3) + 0.5 * theta_hat + (
                (1 / theta ** 2) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * theta_hat @ theta_hat

    @staticmethod
    def jac_left_inverse(theta_vec):
        """Compute the left derivative of Log(X) with respect to X for theta_vec = Log(X).

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        theta = np.linalg.norm(theta_vec)
        if theta < 1e-10:
            return np.identity(3)

        theta_hat = SO3.hat(theta_vec)

        return np.identity(3) - 0.5 * theta_hat + (
                (1 / theta ** 2) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * theta_hat @ theta_hat

    @staticmethod
    def jac_X_oplus_tau_wrt_X(theta_vec):
        """Compute the Jacobian of X.oplus(tau) with respect to the element X

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        return SO3.Exp(theta_vec).inverse().matrix

    @staticmethod
    def jac_X_oplus_tau_wrt_tau(theta_vec):
        """Compute the Jacobian of X.oplus(tau) with respect to the tangent space vector tau

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        return SO3.jac_right(theta_vec)
