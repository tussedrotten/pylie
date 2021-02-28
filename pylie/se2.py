import numpy as np
from pylie import SO2

class SE2:
    """Represents an element of the SO(2) Lie group (rotations in 2D)."""

    def __init__(self, pose_tuple=(SO2(), np.zeros((2, 1)))):
        """Constructs an SO(2) element.
        The default is the identity element theta=0.

        :param angle: The rotation angle in radians (optional).
        """
        self.rotation, self.translation = pose_tuple

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
        if not isinstance(so3, SO2):
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
        if not isinstance(t, np.ndarray) and t.shape == (2, 1):
            raise TypeError('Translation must be a 3D column vector')

        self._translation = t

    def to_matrix(self):
        """Return the matrix representation of this element.

        :return: 2x2 SO(2) matrix
        """
        T = np.identity(3)
        T[0:2, 0:2] = self.rotation.to_matrix()
        T[0:2, 2] = self.translation.T
        return T

    def Log(self):
        """Computes the tangent space vector at the current element X.

        :return: The tangent space vector theta_vec, or angle, axis if split_angle_axis is True.
        """
        return self.angle

    def inverse(self):
        """Compute the inverse of the current element X.

        :return: The inverse of the current element.
        """
        return SO2(-self.angle)

    def action(self, x):
        """Perform the action of the SO(3) element on the 3D column vector x.

        :param x: 3D column vector to be transformed (or a matrix of 3D column vectors)
        :return: The resulting rotated 3D column vectors
        """
        return self.to_matrix @ x

    def compose(self, Y):
        """Compose this element with another element on the right

        :param Y: The other SO3 element
        :return: This element composed with Y
        """
        return SO2(self.angle + Y.angle)

    def adjoint(self):
        """The adjoint at the element.
        :return: The adjoint, a 3x3 rotation matrix.
        """
        return 1.0

    def oplus(self, theta):
        """Computes the right perturbation of Exp(theta_vec) on the element X.

        :param theta_vec: The tangent space vector, a 3D column vector.
        :return: The perturbed SO3 element Y = X :math:`\\oplus` theta_vec.
        """

        return self @ SO2.Exp(theta)

    def ominus(self, X):
        """Computes the tangent space vector at X between X and this element Y.

        :param X: The other element
        :return: The difference theta_vec = Y :math:'\\ominus' X
        """

        return self.angle - X.angle

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
        if isinstance(other, np.ndarray) and other.shape[0] == 2:
            # Other is matrix of 3D column vectors, perform action on vectors.
            return self.action(other)
        else:
            raise TypeError('Argument must be a matrix of 3D column vectors')

    def __matmul__(self, other):
        """Matrix multiplication operator performs composition on elements of SO(3).

        :param other: Other SO3
        :return: Composed SO3
        """
        if isinstance(other, SO2):
            # Other is SO3, perform composition.
            return self.compose(other)
        else:
            raise TypeError('Argument must be an SO3')

    def __len__(self):
        """Length operator returns the dimension of the tangent vector space,
        which is equal to the number of degrees of freedom (DOF).

        :return: The DOF for rotations (3)
        """
        return 3

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

        :param theta_vec: 3d tangent space column vector.
        :return: The Lie Algebra (3x3 matrix).
        """
        return np.array([[0, -theta],
                         [theta, 0]])

    @staticmethod
    def vee(theta_hat):
        """Performs the vee operator on the skew symmetric Lie Algebra matrix theta_hat,
        which returns the corresponding tangent space vector.

        :param theta_hat: The Lie Algebra (3x3 matrix)
        :return: 3d tangent space column vector.
        """
        return theta_hat[1, 0]

    @staticmethod
    def Exp(theta):
        """Computes the Exp-map on the Lie algebra vector theta_vec,
        which transfers it to the corresponding Lie group element.

        :param theta_vec: 3d tangent space column vector.
        :return: Corresponding SO(3) element
        """
        return SO2(theta)
