import visgeom as vg
from pylie import SO3, SE3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

"""Example - Visualizes different perturbations and the path they take along the manifold"""


def vis_perturbations():
    # Define the fixed frame "a" relative to the world frame "w".
    T_w_a = SE3((SO3.from_roll_pitch_yaw(5*np.pi/4, 0, np.pi/2), np.array([[2, 2, 2]]).T))

    # The vector xi represents a perturbation on the tangent vector space.
    # We change the elements in this vector by using the sliders.
    xi_vec = np.zeros([6, 1])

    # We can choose to draw an oriented box around the perturbed pose,
    # and we can draw the trajectory along the manifold (by interpolation).
    draw_options = {'Draw box': False, 'Draw manifold\ntrajectory': True}

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
    xi_sliders = [Slider(plt.axes([0.05, 0.9 - 0.05*i, 0.20, 0.03], facecolor=widget_color),
                         r'$\xi_' + str(i+1) + '$', -4.0, 4.0, valinit=xi_vec[i].item(), valstep=0.01) for i in range(6)]
    button = Button(plt.axes([0.1, 0.55, 0.1, 0.04]), 'Reset', color=widget_color, hovercolor='0.975')
    check = CheckButtons(plt.axes([0.025, 0.10, 0.24, 0.15], facecolor=widget_color),
                         draw_options.keys(),
                         draw_options.values())
    radio = RadioButtons(plt.axes([0.025, 0.3, 0.24, 0.2], facecolor=widget_color),
                         (r'1. $\mathrm{Exp}(\mathbf{\xi})$',
                          r'2. $\mathbf{T}_{wc} \circ \mathrm{Exp}(\mathbf{\xi})$',
                          r'3. $\mathrm{Exp}(\mathbf{\xi}) \circ \mathbf{T}_{wc}$'),
                         active=0)

    # Setup the update callback, which is called by the sliders and the radio buttons.
    def update(val):
        ax.clear()
        for i, slider in enumerate(xi_sliders):
            xi_vec[i] = slider.val

        if radio.value_selected[0] == '1':
            draw_exp(ax, xi_vec, draw_options)
        elif radio.value_selected[0] == '2':
            draw_right_perturbation(ax, T_w_a, xi_vec, draw_options)
        else:
            draw_left_perturbation(ax, T_w_a, xi_vec, draw_options)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-4, 4)
        vg.plot.axis_equal(ax)

        fig.canvas.draw_idle()
    for slider in xi_sliders:
        slider.on_changed(update)
    radio.on_clicked(update)

    # Setup the check buttons to update the "draw options".
    def update_draw_options(label):
        draw_options[label] = not draw_options[label]
        update([])
    check.on_clicked(update_draw_options)

    # Setup the reset callback, used by the reset button.
    def reset(event):
        for slider in xi_sliders:
            slider.reset()
    button.on_clicked(reset)

    # Start with first update.
    update([])
    plt.show()


def draw_exp(ax, xi_vec, draw_options):
    vg.plot_pose(ax, SE3().to_tuple(), scale=1, text='$\mathcal{F}_w$')
    T_l = SE3.Exp(xi_vec)
    vg.plot_pose(ax, T_l.to_tuple()) #, text=r'$\mathrm{Exp}(\mathbf{\xi})$')

    if draw_options['Draw box']:
        box_points = vg.utils.generate_box(pose=T_l.to_tuple(), scale=1)
        vg.utils.plot_as_box(ax, box_points)

    if draw_options['Draw manifold\ntrajectory']:
        draw_interpolated(ax, SE3(), xi_vec, SE3())


def draw_right_perturbation(ax, T_w_a, xi_vec, draw_options):
    vg.plot_pose(ax, SE3().to_tuple(), scale=1, text='$\mathcal{F}_w$')
    vg.plot_pose(ax, T_w_a.to_tuple(), scale=1, text='$\mathcal{F}_a$')
    T_r = T_w_a @ SE3.Exp(xi_vec)

    vg.plot_pose(ax, T_r.to_tuple()) #, text=r'$\mathbf{T}_{wa} \circ \mathrm{Exp}(\mathbf{\xi})$')

    if draw_options['Draw box']:
        box_points = vg.utils.generate_box(pose=T_r.to_tuple(), scale=1)
        vg.utils.plot_as_box(ax, box_points)

    if draw_options['Draw manifold\ntrajectory']:
        draw_interpolated(ax, T_w_a, xi_vec, SE3())


def draw_left_perturbation(ax, T_w_a, xi_vec, draw_options):
    vg.plot_pose(ax, SE3().to_tuple(), scale=1, text='$\mathcal{F}_w$')
    vg.plot_pose(ax, T_w_a.to_tuple(), scale=1, text='$\mathcal{F}_a$')
    T_l = SE3.Exp(xi_vec) @ T_w_a
    vg.plot_pose(ax, T_l.to_tuple()) #, text=r'$\mathrm{Exp}(\mathbf{\xi}) \circ \mathbf{T}_{wa}$')

    if draw_options['Draw box']:
        box_points = vg.utils.generate_box(pose=T_l.to_tuple(), scale=1)
        vg.utils.plot_as_box(ax, box_points)

    if draw_options['Draw manifold\ntrajectory']:
        draw_interpolated(ax, SE3(), xi_vec, T_w_a)


def draw_interpolated(ax, T_1, xi, T_2):
    for alpha in np.linspace(0, 1, 20):
        T = T_1 @ SE3.Exp(alpha * xi) @ T_2
        vg.plot_pose(ax, T.to_tuple(), alpha=0.1)


if __name__ == "__main__":
    vis_perturbations()
