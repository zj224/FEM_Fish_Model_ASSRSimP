import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from Fish_Functions.get_nancy_params import get_vnancy


def animate_eel(
    Y,
    dt,
    nframes=300,
    playback_speed=1.0,
    pad=2.0,
    figsize=(7, 7),
    line_style='-o',
    linewidth=3,
    markersize=4,
    show_time=True,
    show_speed=True,
    save_path=None,
    fps=30,
    show=True,
):
    """
    Animate a planar eel/fish swimming simulation from a state history.

    The body is reconstructed frame-by-frame from the generalized
    coordinates: a head position (x, y), a heading angle theta0, and N
    relative joint angles phi_1..phi_N (N and the segment length L come
    from get_vnancy()). Segment k+1 is placed by walking forward from the
    head at length L, accumulating the joint angles as it goes.

    Parameters
    ----------
    Y : ndarray, shape (nt, 2*nq)
        State history from the simulation
    dt : float
        Simulation time step (seconds)
    nframes : int, default 300
        Number of frames to sample from Y. If Y has fewer rows than this,
        every row is used instead. Lower this for faster previews of long
        simulations; raise it for smoother playback.
    playback_speed : float, default 1.0
        Real-time speed multiplier for the animation. 1.0 plays back at
        the same rate the simulation evolved (e.g. 1 second of sim time
        takes 1 second of wall-clock time); 2.0 plays twice as fast; 0.5
        plays at half speed. Note this is a target — actual playback rate
        is capped by how fast matplotlib can redraw each frame.
    pad : float, default 2.0
        Extra margin (meters) added around the head's trajectory when
        setting the fixed axis limits.
    figsize : tuple, default (7, 7)
        Matplotlib figure size in inches.
    line_style : str, default '-o'
        Matplotlib format string for the body line/markers.
    linewidth : float, default 3
    markersize : float, default 4
    show_time : bool, default True
        If True, overlay the current simulation time (s) on the plot.
    show_speed : bool, default True
        If True, overlay the playback speed multiplier on the plot.
    save_path : str or None, default None
        If given, save the animation to this file instead of (or in
        addition to, if `show=True`) displaying it interactively. Use a
        '.gif' extension to save with Pillow (bundled with matplotlib)
    fps : int, default 30
        Frames per second used when writing `save_path` to file.
    show : bool, default True
        If True, display the animation in an interactive window via
        plt.show(). Set to False on headless machines (e.g. CI, remote
        servers) or when only saving to file.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object. Keep a reference to it (e.g. assign the
        return value to a variable) if the animation stops or fails to
        display — some backends garbage-collect FuncAnimation instances
        that aren't referenced elsewhere.
    """
    VN = get_vnancy()
    N = VN.N
    L = VN.L
    states = Y
    nt = states.shape[0]
    nq = 3 + N

    # Sample a subset of frames for playback/export.
    nframes = min(nframes, nt)
    idxs = np.linspace(0, nt - 1, nframes).astype(int)
    times = idxs * dt

    def body_points(idx):
        """Reconstruct (x, y) of the head and every joint at frame `idx`."""
        st = states[idx]
        q = st[:nq]
        x, y, theta0 = q[0], q[1], q[2]
        phi = q[3:3 + N]

        pts_x = np.zeros(N + 1)
        pts_y = np.zeros(N + 1)
        pts_x[0], pts_y[0] = x, y

        th = theta0
        for k in range(N):
            th += phi[k]
            pts_x[k + 1] = pts_x[k] + L * np.cos(th)
            pts_y[k + 1] = pts_y[k] + L * np.sin(th)
        return pts_x, pts_y

    # --- Setup figure and axis ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal', adjustable='box')

    # Fix axis limits up front (based on the head's full trajectory) so the
    # view doesn't jump around frame to frame.
    xs = states[:, 0]
    ys = states[:, 1]
    xmin, xmax = np.min(xs) - pad, np.max(xs) + pad
    ymin, ymax = np.min(ys) - pad, np.max(ys) + pad

    # Fallback if the head barely moves, so the axes aren't degenerate.
    if xmax - xmin < 1e-3:
        xmin, xmax = -5, 5
    if ymax - ymin < 1e-3:
        ymin, ymax = -5, 5

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Planar Eel (N={N})')

    # Initialize line/text objects once and mutate them each frame, rather
    # than redrawing the axes (much faster, and avoids aspect-ratio resets).
    line, = ax.plot([], [], line_style, linewidth=linewidth, markersize=markersize)
    info_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        ha='left', va='top', fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )

    def init():
        line.set_data([], [])
        info_text.set_text('')
        return line, info_text

    def update(frame_i):
        idx = idxs[frame_i]
        pts_x, pts_y = body_points(idx)
        line.set_data(pts_x, pts_y)

        if show_time or show_speed:
            lines = []
            if show_time:
                lines.append(f't = {times[frame_i]:6.2f} s')
            if show_speed:
                lines.append(f'speed = {playback_speed:g}x')
            info_text.set_text('\n'.join(lines))

        return line, info_text

    # Convert playback_speed into a wall-clock delay between frames: the
    # simulated time between sampled frames, scaled by the desired speed.
    # (idxs is evenly spaced, so any consecutive gap is representative.)
    sim_dt_per_frame = dt * (idxs[1] - idxs[0]) if nframes > 1 else dt
    interval_ms = max(1.0, 1000.0 * sim_dt_per_frame / playback_speed)

    anim = FuncAnimation(
        fig, update, frames=len(idxs), init_func=init,
        interval=interval_ms, blit=True, repeat=False,
    )

    if save_path is not None:
        if save_path.lower().endswith('.mp4'):
            writer = FFMpegWriter(fps=fps)
        else:
            writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return anim
