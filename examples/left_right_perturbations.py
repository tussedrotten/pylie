import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SO3, SE3

"""Example - Perturbations on the left and on the right of an element"""


def left_right_perturbations():
    # Define the pose of a camera.
    T_w_c = SE3((SO3.from_roll_pitch_yaw(5*np.pi/4, 0, np.pi/2), np.array([[2, 2, 2]]).T))

    # Perturbation.
    xsi_vec = np.array([[2, 0, 0, 0, 0, 0]]).T

    # Perform the perturbation on the right (locally).
    T_r = T_w_c @ SE3.Exp(xsi_vec)

    # Perform the perturbation on the left (globally).
    T_l = SE3.Exp(xsi_vec) @ T_w_c

    # We can transform the tangent vector between local and global frames using the adjoint matrix.
    xsi_vec_w = T_w_c.adjoint() @ xsi_vec               # When xsi_vec is used on the right side.
    xsi_vec_c = T_w_c.inverse().adjoint() @ xsi_vec     # When xsi_vec is used on the left side.
    print('xsi_vec_w: ', xsi_vec_w.flatten())
    print('xsi_vec_c: ', xsi_vec_c.flatten())

    # Visualise the perturbations:
    # Use Qt 5 backend in visualisation, use latex.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot frames.
    to_right = np.hstack((T_w_c.translation, T_r.translation))
    ax.plot(to_right[0, :], to_right[1, :], to_right[2, :], 'k:')
    to_left = np.hstack((T_w_c.translation, T_l.translation))
    ax.plot(to_left[0, :], to_left[1, :], to_left[2, :], 'k:')
    vg.plot_pose(ax, SE3().to_tuple(), text='$\mathcal{F}_w$')
    vg.plot_pose(ax, T_w_c.to_tuple(), text='$\mathcal{F}_c$')
    vg.plot_pose(ax, T_r.to_tuple(), text=r'$\mathbf{T}_{wc} \circ \mathrm{Exp}(\mathbf{\xi})$')
    vg.plot_pose(ax, T_l.to_tuple(), text=r'$\mathrm{Exp}(\mathbf{\xi}) \circ \mathbf{T}_{wc}$')

    # Show figure.
    vg.plot.axis_equal(ax)
    plt.show()


if __name__ == "__main__":
    left_right_perturbations()
