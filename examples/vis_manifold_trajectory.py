import visgeom as vg
from pylie import SO3, SE3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

"""Example - Visualises path along the manifold between two poses"""


def vis_manifold_trajectory():
    # The vector theta represents the orientation of the frame "a" on angle-axis form.
    # We change the elements in this vector by using the sliders.
    theta_vec = np.zeros([3, 1])

    # The vector t_w_a represents the position of the frame "a"
    t_w_a = np.array([[1., 0., 0.]]).T

    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    plt.subplots_adjust(left=0.25)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.margins(x=0)

    # Add the widgets.
    widget_color = 'lightgoldenrodyellow'
    theta_sliders = [Slider(plt.axes([0.05, 0.9 - 0.05*i, 0.20, 0.03], facecolor=widget_color),
                         r'$\theta_' + str(i+1) + '$', -4.0, 4.0, valinit=theta_vec[i].item(), valstep=0.01) for i in range(3)]

    t_w_a_sliders = [Slider(plt.axes([0.05, 0.7 - 0.05*i, 0.20, 0.03], facecolor=widget_color),
                         r'$t_' + str(i+1) + '$', -1.0, 1., valinit=t_w_a[i].item(), valstep=0.01) for i in range(3)]

    button = Button(plt.axes([0.1, 0.55, 0.1, 0.04]), 'Reset', color=widget_color, hovercolor='0.975')

    # Set up the update callback, which is called by the sliders and the radio buttons.
    def update(val):
        ax.clear()
        for i, slider in enumerate(theta_sliders):
            theta_vec[i] = slider.val

        for i, slider in enumerate(t_w_a_sliders):
            t_w_a[i] = slider.val

        T_w_a = SE3((SO3.Exp(theta_vec), t_w_a))
        draw_trajectory(ax, T_w_a)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        vg.plot.axis_equal(ax)

        fig.canvas.draw_idle()

    for slider in theta_sliders + t_w_a_sliders:
        slider.on_changed(update)

    # Set up the reset callback, used by the reset button.
    def reset(event):
        for slider in theta_sliders + t_w_a_sliders:
            slider.reset()
    button.on_clicked(reset)

    # Start with first update.
    update([])
    plt.show()


def draw_trajectory(ax, T_w_a):
    vg.plot_pose(ax, SE3().to_tuple(), scale=0.2, text='$\mathcal{F}_w$')
    vg.plot_pose(ax, T_w_a.to_tuple(), scale=0.2, text='$\mathcal{F}_a$')
    draw_interpolated(ax, SE3(), T_w_a)


def draw_interpolated(ax, T_1, T_2):
    for alpha in np.linspace(0, 1, 100):
        T = T_1 + alpha * (T_2 - T_1)
        vg.plot_pose(ax, T.to_tuple(), alpha=0.1, scale=0.2)


if __name__ == "__main__":
    vis_manifold_trajectory()
