import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SE3

"""Example - Estimate the mean pose from a set of random poses"""


def draw_random_poses(mean_pose, cov_pose, n=100):
    """Draw random poses from a pose distribution.
    :param mean_pose: The mean pose, an SE3 object
    :param cov_pose: The covariance, a 6x6 covariance matrix
    :param n: The number of draws
    :return: An array of drawn poses.
    """
    # Create an array of poses, initialised to the mean pose.
    poses = np.full(n, mean_pose, dtype=object)

    # Draw random tangent space vectors.
    random_xis = np.random.multivariate_normal(np.zeros(6), cov_pose, n).T

    # Perturb the mean pose with each of the random tangent space vectors.
    for i in range(n):
        poses[i] = poses[i] + random_xis[0:, [i]]

    return poses


def compute_mean_pose(poses, conv_thresh=1e-14, max_iters=20):
    """Compute the mean pose from an array of poses.
    :param poses: An array of SE3 objects
    :param conv_thresh: The convergence threshold
    :param max_iters: The maximum number of iterations
    :return: The estimate of the mean pose
    """
    num_poses = len(poses)

    # Initialise mean pose.
    mean_pose = poses[0]

    for it in range(max_iters):
        # Compute the mean tangent vector in the tangent space at the current estimate.
        mean_xi = np.zeros((6, 1))
        for pose in poses:
            mean_xi = mean_xi + (pose - mean_pose)
        mean_xi = mean_xi / num_poses

        # Update the estimate.
        mean_pose = mean_pose + mean_xi

        # Stop if the update is small.
        if np.linalg.norm(mean_xi) < conv_thresh:
            break

    return mean_pose


def main():
    # Define the pose distribution.
    mean_pose = SE3()
    cov_pose = np.diag(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.4]) ** 2)

    # Draw random poses from this distribution.
    poses = draw_random_poses(mean_pose, cov_pose)

    # Estimate the mean pose from the random poses.
    estimated_mean_pose = compute_mean_pose(poses)
    print(estimated_mean_pose.to_matrix())

    # Plot result.
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot each of the randomly drawn poses.
    for pose in poses:
        vg.plot_pose(ax, pose.to_tuple(), alpha=0.05)

    # Plot the estimated mean pose.
    vg.plot_pose(ax, estimated_mean_pose.to_tuple())

    # Show figure.
    vg.plot.axis_equal(ax)
    plt.show()


if __name__ == "__main__":
    main()
