import numpy as np
from pylie.common import to_rotation_matrix


class SO2:
    """Represents an element of the SO(2) Lie group (rotations in 2D)."""

    def __init__(self, real=None, imag=None):
        """Constructs a SO(2) element.
        The default is the identity element I.
        The rotation is represented by a unit complex number.

        :param real: The real part of a unit complex number.
        :param imag: The imaginary part of a unit complex number.
        """
        if real is None and imag is None:
            # Default is identity element.
            real = 1.0
            imag = 0.0
        elif real is None or imag is None:
            raise ValueError("Both real and imag arguments must be provided")

        real = np.array(real).item()
        imag = np.array(imag).item()

        self._coeffs = np.array([real, imag])
        self.normalise()

    @classmethod
    def from_angle(cls, angle):
        """Constructs a SO(2) element from a rotation angle.

        :param angle: The rotation angle in radians.
        """
        angle = np.array(angle).item()
        return cls(np.cos(angle), np.sin(angle))

    @classmethod
    def from_matrix(cls, R):
        """Construct a SO(2) element from a matrix.
        The rotation is fitted to the closest rotation matrix

        :param R: 2x2 rotation matrix.
        :return: The SO(2) element.
        """
        R = to_rotation_matrix(R)
        return cls(R[0, 0], R[1, 0])

    def real(self):
        """The real part of the unit complex number representing the rotation.
        :return: The real part.
        """
        return self._coeffs[0]

    def imag(self):
        """The imaginary part of the unit complex number representing the rotation.
        :return: The imaginary part.
        """
        return self._coeffs[1]

    def normalise(self):
        """Normalises the complex number representing the rotation to unit norm."""
        norm = np.linalg.norm(self._coeffs)
        if norm < 1e-16:
            raise ValueError("Norm of representation is almost zero")
        self._coeffs /= norm

    def angle(self):
        """ The angle representation of the SO(2) element

        :return: The angle corresponding to this SO(2) element in radians.
        """
        return np.arctan2(self.imag(), self.real())

    def to_matrix(self):
        """Return the matrix representation of this element.

        :return: 2x2 SO(2) matrix
        """
        cos_theta = self.real()
        sin_theta = self.imag()

        return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    def Log(self):
        """Computes the tangent space vector at the current element X.

        :return: The tangent space vector theta_vec.
        """
        return self.angle()

    def inverse(self):
        """Compute the inverse of the current element X.

        :return: The inverse of the current element.
        """
        return SO2(self.real(), -self.imag())

    def action(self, x):
        """Perform the action of the SO(2) element on the 2D column vector x.

        :param x: 2D column vector to be transformed (or a matrix of 2D column vectors)
        :return: The resulting rotated 2D column vectors
        """
        return self.to_matrix() @ x

    def compose(self, Y):
        """Compose this element with another element on the right

        :param Y: The other SO2 element
        :return: This element composed with Y
        """
        cmp_real = self.real() * Y.real() - self.imag() * Y.imag()
        cmp_imag = self.real() * Y.imag() + self.imag() * Y.real()

        return SO2(cmp_real, cmp_imag)

    def adjoint(self):
        """The adjoint at the element.
        :return: The adjoint, a 1x1 matrix.
        """
        return np.array([[1.0]])

    def oplus(self, theta):
        """Computes the right perturbation of Exp(theta_vec) on the element X.

        :param theta_vec: The tangent space vector, a 1D column vector.
        :return: The perturbed SO2 element Y = X :math:`\\oplus` theta.
        """

        return self @ SO2.Exp(theta)

    def ominus(self, X):
        """Computes the tangent space vector at X between X and this element Y.

        :param X: The other element
        :return: The difference theta_vec = Y :math:'\\ominus' X
        """

        return (X.inverse() @ self).Log()

    def jac_inverse_X_wrt_X(X):
        """Computes the Jacobian of the inverse operation X.inverse() with respect to the element X.

        :return: The Jacobian (1x1 matrix)
        """
        return -np.array([[1.0]])

    def jac_action_Xx_wrt_X(X, x):
        """Computes the Jacobian of the action X.action(x) with respect to the element X.

        :param x: The 2D column vector x.
        :return: The Jacobian (2x1 matrix)
        """
        return X.to_matrix() @ SO2.hat(1.0) @ x

    def jac_action_Xx_wrt_x(X):
        """Computes the Jacobian of the action X.action(x) with respect to the element X.

        :return: The Jacobian (2x2 matrix)
        """
        return X.to_matrix()

    def jac_Y_ominus_X_wrt_X(Y, X):
        """Compute the Jacobian of Y.ominus(X) with respect to the element X.

        :param X: The SO(2) element X.
        :return: The Jacobian (1x1 matrix)
        """
        return -np.array([[1.0]])

    def jac_Y_ominus_X_wrt_Y(Y, X):
        """Compute the Jacobian of Y.ominus(X) with respect to the element Y.

        :param X: The SO(2) element X.
        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])

    def __add__(self, theta):
        """Add operator performs the "oplus" operation on the element X.

        :param theta: The tangent space vector, a 1D column vector.
        :return: The perturbed SO3 element Y = X :math:`\\oplus` theta.
        """
        return self.oplus(theta)

    def __sub__(self, X):
        """Subtract operator performs the "ominus" operation at X between X and this element Y.

        :param X: The other element
        :return: The difference theta_vec = Y :math:'\\ominus' X
        """
        return self.ominus(X)

    def __mul__(self, other):
        """Multiplication operator performs action on vectors.

        :param other: 2D column vector, or a matrix of 2D column vectors
        :return: Transformed 2D column vectors
        """
        if isinstance(other, np.ndarray) and other.shape[0] == 2:
            # Other is matrix of 2D column vectors, perform action on vectors.
            return self.action(other)
        else:
            raise TypeError('Argument must be a matrix of 2D column vectors')

    def __matmul__(self, other):
        """Matrix multiplication operator performs composition on elements of SO(2).

        :param other: Other SO2
        :return: Composed SO2
        """
        if isinstance(other, SO2):
            # Other is SO2, perform composition.
            return self.compose(other)
        else:
            raise TypeError('Argument must be an SO2')

    def __len__(self):
        """Length operator returns the dimension of the tangent vector space,
        which is equal to the number of degrees of freedom (DOF).

        :return: The DOF for SO(2) (1)
        """
        return 1

    def __repr__(self):
        """Formal string representation of the object.

        :return: The formal representation as a string
        """
        return "SO2(\n" + repr(self.to_matrix()) + "\n)"

    def __str__(self):
        """Informal string representation of the object
        prints the matrix representation.

        :return: The matrix representation as a string
        """
        return str(self.to_matrix())


    @staticmethod
    def hat(theta):
        """Performs the hat operator on the tangent space vector theta_vec,
        which returns the corresponding skew symmetric Lie Algebra matrix theta_hat.

        :param theta_vec: 1D tangent space column vector.
        :return: The Lie Algebra (2x2 matrix).
        """
        return np.array([[0, -theta],
                         [theta, 0]])

    @staticmethod
    def vee(theta_hat):
        """Performs the vee operator on the skew symmetric Lie Algebra matrix theta_hat,
        which returns the corresponding tangent space vector.

        :param theta_hat: The Lie Algebra (2x2 matrix)
        :return: 1D tangent space column vector.
        """
        return theta_hat[1, 0].item()

    @staticmethod
    def Exp(theta):
        """Computes the Exp-map on the Lie algebra vector theta_vec,
        which transfers it to the corresponding Lie group element.

        :param theta: 1D tangent space column vector.
        :return: Corresponding SO(2) element
        """
        return SO2.from_angle(theta)

    @staticmethod
    def jac_composition_XY_wrt_X(Y):
        """Computes the Jacobian of the composition X.compose(Y) with respect to the element X.

        :param Y: SO3 element Y
        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])

    @staticmethod
    def jac_composition_XY_wrt_Y():
        """Computes the Jacobian of the composition X.compose(Y) with respect to the element Y.

        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])

    @staticmethod
    def jac_right(theta_vec):
        """Compute the right derivative of Exp(theta_vec) with respect to theta_vec.

        :param theta_vec: The tangent space 1D column vector.
        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])

    @staticmethod
    def jac_left(theta_vec):
        """Compute the left derivative of Exp(theta_vec) with respect to theta_vec.

        :param theta_vec: The tangent space 1D column vector.
        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])

    @staticmethod
    def jac_right_inverse(theta_vec):
        """Compute the right derivative of Log(X) with respect to X for theta_vec = Log(X).

        :param theta_vec: The tangent space 1D column vector.
        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])

    @staticmethod
    def jac_left_inverse(theta_vec):
        """Compute the left derivative of Log(X) with respect to X for theta_vec = Log(X).

        :param theta_vec: The tangent space 1D column vector.
        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])

    @staticmethod
    def jac_X_oplus_tau_wrt_X(theta_vec):
        """Compute the Jacobian of X.oplus(tau) with respect to the element X

        :param theta_vec: The tangent space 1D column vector.
        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])

    @staticmethod
    def jac_X_oplus_tau_wrt_tau(theta_vec):
        """Compute the Jacobian of X.oplus(tau) with respect to the tangent space vector tau

        :param theta_vec: The tangent space 1D column vector.
        :return: The Jacobian (1x1 matrix)
        """
        return np.array([[1.0]])
