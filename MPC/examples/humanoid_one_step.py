"""Humanoid planning to walk a single step ahead, with stick-figure animation."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from qpmpc import MPCProblem, solve_mpc


@dataclass
class Parameters:
    """Parameters of the step and humanoid."""

    com_height: float = 0.8
    dsp_duration: float = 0.1  # [s]
    end_pos: float = 0.3  # [m]
    foot_length: float = 0.1  # [m]
    gravity: float = 9.81  # [m] / [s]²
    horizon_duration: float = 2.5  # [s]
    nb_timesteps: int = 16
    ssp_duration: float = 0.7  # [s]
    start_pos: float = 0.0  # [m]


def build_mpc_problem(params: Parameters):
    """Build the model predictive control problem.

    For details on this problem and how model predictive control can be used
    for humanoid stepping, see "Trajectory free linear model predictive control
    for stable walking in the presence of strong perturbations" (Wieber, 2006).
    """
    T = params.horizon_duration / params.nb_timesteps
    nb_init_dsp_steps = int(round(params.dsp_duration / T))
    nb_init_ssp_steps = int(round(params.ssp_duration / T))
    nb_dsp_steps = int(round(params.dsp_duration / T))

    state_matrix = np.array(
        [[1.0, T, T**2 / 2.0],
         [0.0, 1.0, T],
         [0.0, 0.0, 1.0]]
    )
    input_matrix = np.array([T**3 / 6.0, T**2 / 2.0, T]).reshape((3, 1))

    eta = params.com_height / params.gravity
    zmp_from_state = np.array([1.0, 0.0, -eta])
    ineq_matrix = np.array([+zmp_from_state, -zmp_from_state])

    cur_max = params.start_pos + 0.5 * params.foot_length
    cur_min = params.start_pos - 0.5 * params.foot_length
    next_max = params.end_pos + 0.5 * params.foot_length
    next_min = params.end_pos - 0.5 * params.foot_length

    ineq_vector = []
    for i in range(params.nb_timesteps):
        if i < nb_init_dsp_steps:
            vec = np.array([+1000.0, +1000.0])  # no constraint
        elif i - nb_init_dsp_steps <= nb_init_ssp_steps:
            vec = np.array([+cur_max, -cur_min])  # around current foot
        elif i - nb_init_dsp_steps - nb_init_ssp_steps < nb_dsp_steps:
            vec = np.array([+1000.0, +1000.0])  # no constraint
        else:
            vec = np.array([+next_max, -next_min])  # around next foot
        ineq_vector.append(vec)

    return MPCProblem(
        transition_state_matrix=state_matrix,
        transition_input_matrix=input_matrix,
        ineq_state_matrix=ineq_matrix,
        ineq_input_matrix=None,
        ineq_vector=ineq_vector,
        initial_state=np.array([params.start_pos, 0.0, 0.0]),
        goal_state=np.array([params.end_pos, 0.0, 0.0]),
        nb_timesteps=params.nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-3,
    )


def plot_plan(params, mpc_problem, plan):
    """Plot CoM and ZMP vs time."""
    t = np.linspace(0.0, params.horizon_duration, params.nb_timesteps + 1)
    X = plan.states
    eta = params.com_height / params.gravity
    zmp_from_state = np.array([1.0, 0.0, -eta])
    zmp = X.dot(zmp_from_state)
    pos = X[:, 0]

    zmp_min = [
        x[0] if abs(x[0]) < 10 else None for x in mpc_problem.ineq_vector
    ]
    zmp_max = [
        -x[1] if abs(x[1]) < 10 else None for x in mpc_problem.ineq_vector
    ]
    # 마지막 스텝에도 동일한 bound 유지
    zmp_min.append(zmp_min[-1])
    zmp_max.append(zmp_max[-1])

    plt.figure()
    plt.plot(t, pos, label="CoM x")
    plt.plot(t, zmp, "r-", label="ZMP")
    plt.plot(t, zmp_min, "g:", label="ZMP min")
    plt.plot(t, zmp_max, "b:", label="ZMP max")
    plt.xlabel("time [s]")
    plt.ylabel("x [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def build_feet_trajectories(params: Parameters, N: int):
    """Compute simple stance/swing foot x trajectories over N+1 states."""
    T = params.horizon_duration / params.nb_timesteps
    nb_init_dsp_steps = int(round(params.dsp_duration / T))
    nb_init_ssp_steps = int(round(params.ssp_duration / T))
    nb_dsp_steps = int(round(params.dsp_duration / T))

    stance_x = np.zeros(N + 1)
    swing_x = np.zeros(N + 1)

    for k in range(N + 1):
        if k <= nb_init_dsp_steps:
            # both feet at start
            stance_x[k] = params.start_pos
            swing_x[k] = params.start_pos
        elif k <= nb_init_dsp_steps + nb_init_ssp_steps:
            # SSP: stance at start, swing moves to end
            alpha = (k - nb_init_dsp_steps) / float(nb_init_ssp_steps)
            stance_x[k] = params.start_pos
            swing_x[k] = params.start_pos + alpha * (params.end_pos - params.start_pos)
        else:
            # final DSP: both at end
            stance_x[k] = params.end_pos
            swing_x[k] = params.end_pos

    return stance_x, swing_x


def animate_humanoid(params: Parameters, plan):
    """Stick-figure animation of the humanoid walking one step."""

    X = plan.states  # shape (N+1, 3)
    N = X.shape[0] - 1
    com_x = X[:, 0]
    hip_z = params.com_height

    stance_x, swing_x = build_feet_trajectories(params, N)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_aspect("equal")
    ax.set_xlim(params.start_pos - 0.2, params.end_pos + 0.4)
    ax.set_ylim(-0.1, hip_z + 0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Humanoid one-step MPC (stick figure)")

    # ground
    ground, = ax.plot(
        [params.start_pos - 0.5, params.end_pos + 0.5],
        [0.0, 0.0],
        "k-", linewidth=4,
    )

    # stance leg (blue), swing leg (red), hip marker (black)
    stance_leg, = ax.plot([], [], "b-", linewidth=4)
    swing_leg, = ax.plot([], [], "r-", linewidth=4)
    hip_point, = ax.plot([], [], "ko", markersize=6)

    def init():
        stance_leg.set_data([], [])
        swing_leg.set_data([], [])
        hip_point.set_data([], [])
        return stance_leg, swing_leg, hip_point

    def update(frame):
        # positions at this frame
        hx = com_x[frame]
        hz = hip_z
        sx = stance_x[frame]
        swx = swing_x[frame]

        # stance leg: stance foot -> hip
        stance_leg.set_data([sx, hx], [0.0, hz])

        # swing leg: hip -> swing foot
        swing_leg.set_data([hx, swx], [hz, 0.0])

        # hip point
        hip_point.set_data([hx], [hz])

        return stance_leg, swing_leg, hip_point

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=N + 1,
        interval=100,  # ms
        blit=True,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    params = Parameters()
    mpc_problem = build_mpc_problem(params)

    # proxqp 사용 (proxsuite 설치되어 있어야 함)
    plan = solve_mpc(mpc_problem, solver="proxqp")

    # 1) CoM / ZMP 플롯
    plot_plan(params, mpc_problem, plan)

    # 2) 스틱 피겨 humanoid 애니메이션
    animate_humanoid(params, plan)
