"""
Diagnostic plots for a simulated fish state history.

All functions here take `state_history`, the (nt, 2*num_generalized_coords)
array produced by main_fish.py's simulation loop, where each row is
[x, y, theta0, phi_1, ..., phi_N, xdot, ydot, theta0dot, phidot_1, ..., phidot_N].
The number of joints N is inferred from the array's width, so no extra
parameters object is required.
"""
import numpy as np
import matplotlib.pyplot as plt


def _split_state(state_history):
    """Split a state history array into position (q) and velocity (qdot) halves."""
    num_generalized_coords = state_history.shape[1] // 2
    q = state_history[:, :num_generalized_coords]
    qdot = state_history[:, num_generalized_coords:]
    return q, qdot


def plot_xy_tau(time_points, state_history, control_history, show=True):
    """
    Plot head x position, head y position, and the applied control input,
    each vs. time, stacked in one figure.

    Parameters
    ----------
    time_points : ndarray
        Simulation time stamps. May be one entry longer than
        state_history/control_history (as produced by main_fish.py, whose
        loop logs state before advancing); the extra entry is dropped.
    state_history : ndarray, shape (nt, 2*num_generalized_coords)
        Logged state history; columns 0 and 1 are head x, y.
    control_history : ndarray, shape (nt,)
        Logged control input applied at each time step.
    show : bool, default True
        If True, call plt.show(). Set False to keep composing the figure
        (e.g. saving to file) without opening an interactive window.

    Returns
    -------
    fig, axes : the created figure and its 3 subplots (x, y, control).
    """
    nt = len(state_history)
    t = time_points[:nt]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, state_history[:, 0])
    axes[0].set_ylabel('x position (m)')
    axes[0].set_title('Head Position and Control Input Over Time')
    axes[0].grid(True)

    axes[1].plot(t, state_history[:, 1])
    axes[1].set_ylabel('y position (m)')
    axes[1].grid(True)

    axes[2].plot(t, control_history[:nt])
    axes[2].set_ylabel('control input')
    axes[2].set_xlabel('time (s)')
    axes[2].grid(True)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_trajectory(state_history, color_by_time=True, show=True):
    """
    Plot the head's path in the x-y plane, marking the start and end.

    Parameters
    ----------
    state_history : ndarray, shape (nt, 2*num_generalized_coords)
        Logged state history; columns 0 and 1 are head x, y.
    color_by_time : bool, default True
        If True, color the path by elapsed simulation step using a
        colormap, so you can see where the fish moved quickly (spread out
        points) vs. slowly/lingered (bunched points), and the direction of
        travel without relying only on the start/end markers.
    show : bool, default True
        If True, call plt.show().

    Returns
    -------
    fig, ax : the created figure and axis.
    """
    x, y = state_history[:, 0], state_history[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    if color_by_time:
        sc = ax.scatter(x, y, c=np.arange(len(x)), cmap='viridis', s=8)
        fig.colorbar(sc, ax=ax, label='time step')
    else:
        ax.plot(x, y, 'b-', linewidth=2)

    ax.plot(x[0], y[0], 'go', markersize=10, label='start', zorder=3)
    ax.plot(x[-1], y[-1], 'ro', markersize=10, label='end', zorder=3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Head Trajectory')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()

    if show:
        plt.show()
    return fig, ax


def plot_joint_angles(time_points, state_history, show=True):
    """
    Plot every joint angle phi_i vs. time, one line per joint.

    Useful for checking that the traveling-wave gait looks right: the
    joints should oscillate with a fixed phase lag from head to tail.

    Parameters
    ----------
    time_points : ndarray
        Simulation time stamps (see plot_xy_tau for length handling).
    state_history : ndarray, shape (nt, 2*num_generalized_coords)
    show : bool, default True

    Returns
    -------
    fig, ax : the created figure and axis.
    """
    q, _ = _split_state(state_history)
    num_joints = q.shape[1] - 3
    t = time_points[:len(state_history)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(num_joints):
        ax.plot(t, np.degrees(q[:, 3 + i]), label=f'phi_{i + 1}')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('joint angle (deg)')
    ax.set_title('Joint Angles Over Time')
    ax.grid(True)
    ax.legend(ncol=min(num_joints, 6), fontsize=8)

    if show:
        plt.show()
    return fig, ax


def plot_heading(time_points, state_history, show=True):
    """
    Plot the head heading angle theta0 vs. time.

    Parameters
    ----------
    time_points : ndarray
    state_history : ndarray, shape (nt, 2*num_generalized_coords)
    show : bool, default True

    Returns
    -------
    fig, ax : the created figure and axis.
    """
    t = time_points[:len(state_history)]
    theta0_deg = np.degrees(state_history[:, 2])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, theta0_deg)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('heading theta0 (deg)')
    ax.set_title('Head Heading Over Time')
    ax.grid(True)

    if show:
        plt.show()
    return fig, ax


def plot_speed(time_points, state_history, show=True):
    """
    Plot the head's translational speed vs. time, computed from the
    logged velocity state (not a finite-difference estimate), plus its
    running average to show whether the swimmer has settled into a
    steady cruising speed.

    Parameters
    ----------
    time_points : ndarray
    state_history : ndarray, shape (nt, 2*num_generalized_coords)
    show : bool, default True

    Returns
    -------
    fig, ax : the created figure and axis.
    """
    _, qdot = _split_state(state_history)
    speed = np.hypot(qdot[:, 0], qdot[:, 1])
    running_avg = np.cumsum(speed) / np.arange(1, len(speed) + 1)
    t = time_points[:len(state_history)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, speed, label='instantaneous speed', alpha=0.6)
    ax.plot(t, running_avg, label='running average', linewidth=2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('head speed (m/s)')
    ax.set_title('Head Speed Over Time')
    ax.grid(True)
    ax.legend()

    if show:
        plt.show()
    return fig, ax
