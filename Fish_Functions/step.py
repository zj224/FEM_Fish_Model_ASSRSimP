import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

@njit
def grouped_traveling_wave_torque_numba(t, N, G, A_group, freq, C_turn, tau_opt):
    """
    Returns torque vector of length N (one torque per joint phi_i).
    Evenly distribute torque across groups (G groups of size N/G).
    """
    tau = np.zeros(N)
    group_size = N // G
    omega = 2.0 * np.pi * freq
    k_wave = 2.0 * np.pi / G
    # compute group torques
    for g in range(G):
        tg = A_group * np.sin(omega * t - k_wave * g) + tau_opt*C_turn
        start = g * group_size
        end = start + group_size
        for i in range(start, end):
            tau[i] = tg / group_size
    # small envelope taper (head->tail)
    # create linear envelope [0.7 .. 1.0]
    for i in range(N):
        env = 0.7 + 0.3 * (i / (N - 1))
        tau[i] = tau[i] * env
    return tau

@njit
def mass_matrix_and_jacobians_numba(q, N, L, m, Izz):
    """
    Compute full mass matrix M (nq x nq), per-link Jv (N,2,nq), per-link Jw (N,nq)
    q is length nq = 3 + N (x,y,theta0, phi1..phiN)
    Returns:
        M : (nq, nq)
        Jv_stack: (N, 2, nq)
        Jw_stack: (N, nq)
        theta_abs: (N,)
    This is optimized with cumulative sums for partial sums of cos/sin.
    """
    nq = 3 + N
    x = q[0]; y = q[1]; theta0 = q[2]
    phi = q[3:3+N]

    theta_abs = np.empty(N)
    # theta_abs[k] = theta0 + sum_{j=0..k} phi[j]
    s = 0.0
    for k in range(N):
        s += phi[k]
        theta_abs[k] = theta0 + s

    # cumsums of cos and sin for theta_abs for quick range sums
    cos_t = np.empty(N)
    sin_t = np.empty(N)
    for i in range(N):
        cos_t[i] = np.cos(theta_abs[i])
        sin_t[i] = np.sin(theta_abs[i])

    cumsum_cos = np.empty(N + 1)
    cumsum_sin = np.empty(N + 1)
    cumsum_cos[0] = 0.0
    cumsum_sin[0] = 0.0
    for i in range(N):
        cumsum_cos[i+1] = cumsum_cos[i] + cos_t[i]
        cumsum_sin[i+1] = cumsum_sin[i] + sin_t[i]

    nq_local = nq
    M = np.zeros((nq_local, nq_local))
    Jv_stack = np.zeros((N, 2, nq_local))
    Jw_stack = np.zeros((N, nq_local))

    for k in range(N):
        # Jacobian Jv for COM of link k (index k)
        # columns:
        # 0: dr/dx = [1;0]
        # 1: dr/dy = [0;1]
        # 2: derivative wrt theta0: sum_{j=0..k-1} L * [-sin(theta_abs[j]); cos(theta_abs[j])] + L/2 * [-sin(theta_abs[k]); cos(theta_abs[k])]
        Jv = np.zeros((2, nq_local))
        Jv[0, 0] = 1.0
        Jv[1, 1] = 1.0

        # contribution from theta0:
        # sum_sin = sum_{j=0..k-1} sin_t[j] = cumsum_sin[k]
        # NOTE: if k==0 -> sum is zero
        sum_sin_before = cumsum_sin[k]    # sum sin up to index k-1
        sum_cos_before = cumsum_cos[k]
        # d/dtheta0
        Jv[0, 2] = -L * sum_sin_before - 0.5 * L * sin_t[k]
        Jv[1, 2] =  L * sum_cos_before + 0.5 * L * cos_t[k]

        # derivatives wrt phi_i (each internal joint)
        # For column corresponding to phi_i (i from 0..N-1) it contributes when i <= k
        for i in range(N):
            if i <= k:
                # sum_{j=i..k-1} sin_t[j] = cumsum_sin[k] - cumsum_sin[i]
                ssin = cumsum_sin[k] - cumsum_sin[i]
                scos = cumsum_cos[k] - cumsum_cos[i]
                Jv[0, 3 + i] = -L * ssin - 0.5 * L * sin_t[k]
                Jv[1, 3 + i] =  L * scos + 0.5 * L * cos_t[k]
            else:
                Jv[0, 3 + i] = 0.0
                Jv[1, 3 + i] = 0.0

        # Angular Jacobian: Jw: theta_dot = theta0_dot + sum_{j=0..k} phi_dot_j
        Jw = np.zeros(nq_local)
        Jw[2] = 1.0
        for j in range(k + 1):
            Jw[3 + j] = 1.0

        # store
        for a in range(nq_local):
            Jv_stack[k, 0, a] = Jv[0, a]
            Jv_stack[k, 1, a] = Jv[1, a]
            Jw_stack[k, a] = Jw[a]

        # mass contribution
        # M += m * Jv^T Jv + Izz * Jw^T Jw
        # compute Jv^T Jv and Jw^T Jw
        # Jv^T Jv (nq x nq)
        for i_col in range(nq_local):
            for j_col in range(nq_local):
                M[i_col, j_col] += m * (Jv[0, i_col] * Jv[0, j_col] + Jv[1, i_col] * Jv[1, j_col]) + Izz * (Jw[i_col] * Jw[j_col])

    return M, Jv_stack, Jw_stack, theta_abs

@njit
def eel_eom_numba(t, y, tau, VN):
#def eel_eom_numba(t, y, tau, params_arr):
    """
    Numba-friendly dynamics function for MPC.
    params_arr is a 1D array with scalars in fixed positions:
    [N, L, m, Izz, g, c_t, c_n, joint_k, joint_b, G, A_group, freq, C_turn]
    C_turn_value is the control input that overrides params_arr[12]
    """
    # unpack
    N = VN.N
    L = VN.L
    m = VN.m
    Izz = VN.Izz
    c_t = VN.ct
    c_n = VN.cn
    joint_k = VN.joint_k
    joint_b = VN.joint_b
    G = VN.G
    A_group = VN.A_group
    freq = VN.freq
    C_turn = VN.C_turn

    # unpack
    #N = int(params_arr[0])
    #L = params_arr[1]
    #m = params_arr[2]
    #Izz = params_arr[3]
    #c_t = params_arr[5]
    #c_n = params_arr[6]
    #joint_k = params_arr[7]
    #joint_b = params_arr[8]
    #G = int(params_arr[9])
    #A_group = params_arr[10]
    #freq = params_arr[11]
    #C_turn = params_arr[12]


    nq = 3 + N
    q = y[:nq]
    qdot = y[nq:nq*2]

    # compute Mass matrix and jacobians
    M, Jv_stack, Jw_stack, theta_abs = mass_matrix_and_jacobians_numba(q, N, L, m, Izz)

    # regularize M slightly (tiny diag)
    traceM = 0.0
    for i in range(nq):
        traceM += M[i, i]
    eps_reg = 1e-8 * (traceM / max(1.0, float(nq)))
    for i in range(nq):
        M[i, i] += eps_reg

    # Hydrodynamic drag Qh (quadratic)
    Qh = np.zeros(nq)
    # For each link compute vk = Jv * qdot (2-vector)
    for k in range(N):
        # Jv_stack[k] shape (2, nq)
        vkx = 0.0
        vky = 0.0
        for col in range(nq):
            vkx += Jv_stack[k, 0, col] * qdot[col]
            vky += Jv_stack[k, 1, col] * qdot[col]
        th = theta_abs[k]
        t_hat_x = np.cos(th)
        t_hat_y = np.sin(th)
        n_hat_x = -np.sin(th)
        n_hat_y = np.cos(th)
        vt = vkx * t_hat_x + vky * t_hat_y
        vn = vkx * n_hat_x + vky * n_hat_y
        Ft_x = -c_t * vt * abs(vt) * t_hat_x
        Ft_y = -c_t * vt * abs(vt) * t_hat_y
        Fn_x = -c_n * vn * abs(vn) * n_hat_x
        Fn_y = -c_n * vn * abs(vn) * n_hat_y
        Fhyd_x = Ft_x + Fn_x
        Fhyd_y = Ft_y + Fn_y
        # project to generalized forces Qh += Jv^T * Fhyd
        for i in range(nq):
            Qh[i] += Jv_stack[k, 0, i] * Fhyd_x + Jv_stack[k, 1, i] * Fhyd_y

    # Joint torques and passive torques (use C_turn_value instead of params_arr[12])
    tau_joints = grouped_traveling_wave_torque_numba(t, N, G, A_group, freq, C_turn, tau)
    phi = q[3:3+N]
    phidot = qdot[3:3+N]
    tau_passive = np.zeros(N)
    for i in range(N):
        tau_passive[i] = -joint_k * phi[i] - joint_b * phidot[i]
    tau_total = np.zeros(N)
    for i in range(N):
        tau_total[i] = tau_joints[i] + tau_passive[i]

    Qtau = np.zeros(nq)
    for i in range(N):
        Qtau[3 + i] = tau_total[i]

    Q = Qh + Qtau

    # Solve for accelerations qddot via linear solve M qddot = Q
    qddot = np.linalg.solve(M, Q)

    # ydot
    ydot = np.zeros(nq * 2)
    for i in range(nq):
        ydot[i] = qdot[i]
        ydot[nq + i] = qddot[i]
    return ydot

def dynamics_py(t, y, tau_opt, VN):
    # wrapper to call compiled numba dynamics
    return eel_eom_numba(t, y, tau_opt, VN)

def step(y0_vec, t0, dt, tau_C, VN):
    """
    Step the system forward by one time step using the given control input.
    """
    tspan = np.array([t0, t0 + dt])
    t_points = np.array([t0, t0 + dt])
    
    sol = solve_ivp(fun=lambda t, y: dynamics_py(t, y, tau_C, VN),
                    t_span=tspan, y0=y0_vec, t_eval=t_points,
                    method='BDF', vectorized=False,
                    rtol=1e-1, atol=1e-3)
    
    # Return the final state
    return sol.y[:, -1]