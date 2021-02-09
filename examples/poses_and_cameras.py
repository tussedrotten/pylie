import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SO3, SE3

"""Example - Simple example with pose compositions and point transformations"""


def poses_and_cameras():
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot the pose of the world North-East-Down (NED) frame (relative to the world frame).
    T_w_w = SE3()
    vg.plot_pose(ax, T_w_w.to_tuple(), scale=3, text='$\mathcal{F}_w$')

    # Plot the body frame (a body-fixed Forward-Right-Down (FRD) frame).
    roll = np.radians(-10)
    pitch = np.radians(0)
    yaw = np.radians(135)
    t_w_b = np.array([[-10, -10, -2]]).T
    T_w_b = SE3((SO3.from_roll_pitch_yaw(roll, pitch, yaw), t_w_b))
    vg.plot_pose(ax, T_w_b.to_tuple(), scale=3, text='$\mathcal{F}_b$')

    # Plot the camera frame.
    # The camera is placed 2 m directly above the body origin.
    # Its optical axis points to the left (in opposite direction of the y-axis in F_b).
    # Its y-axis points downwards along the z-axis of F_b.
    R_b_c = np.array([[1, 0, 0],
                      [0, 0, -1],
                      [0, 1, 0]])
    t_b_c = np.array([[0, 0, -2]]).T
    T_b_c = SE3((SO3(R_b_c), t_b_c))
    T_w_c = T_w_b @ T_b_c
    vg.plot_pose(ax, T_w_c.to_tuple(), scale=3, text='$\mathcal{F}_c$')

    # Plot obstacle frame.
    # The cube is placed at (North: 10 m, East: 10 m, Down: -1 m).
    # Its top points upwards, and its front points south.
    R_w_o = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])
    t_w_o = np.array([[10, 10, -1]]).T
    T_w_o = SE3((SO3(R_w_o), t_w_o))
    vg.plot_pose(ax, T_w_o.to_tuple(), scale=3, text='$\mathcal{F}_o$')

    # Plot the cube with sides 3 meters.
    points_o = vg.utils.generate_box(scale=3)
    points_w = T_w_o * points_o
    vg.utils.plot_as_box(ax, points_w)

    # Plot the image plane.
    img_plane_scale = 3
    K = np.array([[50, 0, 40],
                  [0, 50, 30],
                  [0, 0, 1]])
    vg.plot_camera_image_plane(ax, K, T_w_c.to_tuple(), scale=img_plane_scale)

    # Project the box onto the normalised image plane (at z=img_plane_scale).
    points_c = T_w_c.inverse() @ T_w_o * points_o
    xn = points_c / points_c[2, :]
    xn_w = T_w_c * (img_plane_scale * xn)
    vg.utils.plot_as_box(ax, xn_w)

    # Show figure.
    vg.plot.axis_equal(ax)
    ax.invert_zaxis()
    ax.invert_yaxis()
    plt.show()


if __name__ == "__main__":
    poses_and_cameras()
