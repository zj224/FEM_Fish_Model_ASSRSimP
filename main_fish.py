"""
Simulate a planar N-link eel/fish swimming under a traveling-wave joint
torque pattern, then animate the resulting trajectory.

State layout (see Fish_Functions/step.py for the full dynamics):
    q    = [x, y, theta0, phi_1, ..., phi_N]   generalized coordinates
    qdot = [xdot, ydot, theta0dot, phidot_1, ..., phidot_N]
    state = [q, qdot]                          shape (2 * num_generalized_coords,)

where (x, y) is the head position, theta0 is the head heading, and
phi_1..phi_N are the relative joint angles between the N body segments.
"""
import numpy as np
from pytictoc import TicToc

from Fish_Functions.get_nancy_params import get_vnancy
from Fish_Functions.step import step
from Fish_Functions.animate_eel import animate_eel
from Fish_Functions.plot import plot_xy_tau, plot_trajectory, plot_joint_angles, plot_heading, plot_speed

# Body/hydrodynamic parameters for the fish model (segment count, length,
# mass, drag coefficients, joint stiffness/damping, gait parameters, etc.).
# See get_nancy_params.get_vnancy for defaults and units.
fish_params = get_vnancy()

sim_timer = TicToc()
sim_timer.tic()

# ---------- TIME SETUP ----------
sim_time_span = np.array([0.0, 1.0])   # [start, end] simulation time, seconds
dt = 0.02                              # integration time step, seconds
time_points = np.arange(sim_time_span[0], sim_time_span[1], dt)

# ---------- INITIAL STATE ----------
initial_x, initial_y, initial_theta0 = 0.0, 0.0, 0.0
initial_phi = np.zeros(fish_params.N)  # body starts straight (no joint bend)
initial_q = np.concatenate(([initial_x, initial_y, initial_theta0], initial_phi))

num_generalized_coords = 3 + fish_params.N  # x, y, theta0 + one angle per joint
initial_qdot = np.zeros(num_generalized_coords)  # starts at rest
initial_state = np.concatenate((initial_q, initial_qdot))

# Turning control input (C_turn override) applied to the traveling-wave
# gait every step. See grouped_traveling_wave_torque_numba in step.py for
# how this biases the gait to turn the fish.
turn_control = 0.5

# ---------- SIMULATION ----------
# History buffers: one entry appended per time step, converted to arrays
# after the loop for plotting/analysis.
state_history = []
control_history = []

state = initial_state
print(f"Starting simulation with {len(time_points)} time steps...")

for step_idx in range(len(time_points) - 1):  # -1 so the final step doesn't run past sim_time_span[1]
    t_current = time_points[step_idx]

    # Log the state/control *before* stepping, so state_history[0] is the
    # initial condition and has the same length as time_points[:-1].
    state_history.append(state.copy())
    control_history.append(turn_control)

    # Advance the dynamics by one time step under the current control input.
    state = step(state, t_current, dt, turn_control, fish_params)

print(f"Simulation complete! Total steps: {len(state_history)}")
sim_timer.toc("Complete simulation took:")

# Convert history lists to arrays for analysis/plotting.
state_history = np.array(state_history)
control_history = np.array(control_history)


# ---------- OUTPUT ----------
# Animate the resulting trajectory and joint angles
animate_eel(state_history, dt=dt)

# Diagnostic plots:
plot_xy_tau(time_points, state_history, control_history)
plot_trajectory(state_history)
#plot_joint_angles(time_points, state_history)
plot_heading(time_points, state_history)
#plot_speed(time_points, state_history)


if __name__ == "__main__":
    pass
