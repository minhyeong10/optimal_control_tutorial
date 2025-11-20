from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from qpmpc import MPCProblem, solve_mpc


@dataclass
class Parameters:
    """Parameters of the step and humanoid (for ONE step)."""

    com_height: float = 0.8
    dsp_duration: float = 0.1  # [s]
    end_pos: float = 0.3       # [m] single step length (from start_pos)
    foot_length: float = 0.1   # [m]
    gravity: float = 9.81      # [m] / [s]²
    horizon_duration: float = 2.5  # [s] per step
    nb_timesteps: int = 16
    ssp_duration: float = 0.7  # [s]
    start_pos: float = 0.0     # [m]


def build_mpc_problem(params: Parameters, initial_state=None):
    """Build MPC problem for ONE step (ZMP in [current, next] foot)."""

    T = params.horizon_duration / params.nb_timesteps
    nb_init_dsp_steps = int(round(params.dsp_duration / T))
    nb_init_ssp_steps = int(round(params.ssp_duration / T))
    nb_dsp_steps = int(round(params.dsp_duration / T))
    _ = nb_dsp_steps  # not used explicitly but left for clarity

    # LIPM state and input matrices
    A = np.array(
        [[1.0, T, T**2 / 2.0],
         [0.0, 1.0, T],
         [0.0, 0.0, 1.0]]
    )
    B = np.array([T**3 / 6.0, T**2 / 2.0, T]).reshape((3, 1))

    eta = params.com_height / params.gravity
    zmp_from_state = np.array([1.0, 0.0, -eta])
    C_ineq = np.array([+zmp_from_state, -zmp_from_state])

    cur_max = params.start_pos + 0.5 * params.foot_length
    cur_min = params.start_pos - 0.5 * params.foot_length
    next_max = params.end_pos + 0.5 * params.foot_length
    next_min = params.end_pos - 0.5 * params.foot_length

    ineq_vector = []
    for i in range(params.nb_timesteps):
        if i < nb_init_dsp_steps:
            # no constraint in first DSP
            vec = np.array([+1000.0, +1000.0])
        elif i - nb_init_dsp_steps <= nb_init_ssp_steps:
            # SSP: ZMP around current foot
            vec = np.array([+cur_max, -cur_min])
        else:
            # later DSP: ZMP around next foot
            vec = np.array([+next_max, -next_min])
        ineq_vector.append(vec)

    if initial_state is None:
        initial_state = np.array([params.start_pos, 0.0, 0.0])

    return MPCProblem(
        transition_state_matrix=A,
        transition_input_matrix=B,
        ineq_state_matrix=C_ineq,
        ineq_input_matrix=None,
        ineq_vector=ineq_vector,
        initial_state=np.asarray(initial_state),
        goal_state=np.array([params.end_pos, 0.0, 0.0]),
        nb_timesteps=params.nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-3,
    )


def build_step_feet_trajectories(params: Parameters,
                                 N: int,
                                 left_init: float,
                                 right_init: float,
                                 stance_is_left: bool):
    """
    한 스텝 동안 왼발/오른발 x궤적 생성.

    - stance_is_left == True  : 왼발이 지지발, 오른발이 스윙
    - stance_is_left == False : 오른발이 지지발, 왼발이 스윙
    """

    T = params.horizon_duration / params.nb_timesteps
    nb_init_dsp_steps = int(round(params.dsp_duration / T))
    nb_init_ssp_steps = int(round(params.ssp_duration / T))
    nb_dsp_steps = int(round(params.dsp_duration / T))
    _ = nb_dsp_steps

    left_x = np.zeros(N + 1)
    right_x = np.zeros(N + 1)

    if stance_is_left:
        stance_init = left_init
        swing_init = right_init
        swing_final = params.end_pos   # 오른발 착지 위치
        left_final = stance_init
        right_final = swing_final
    else:
        stance_init = right_init
        swing_init = left_init
        swing_final = params.end_pos   # 왼발 착지 위치
        right_final = stance_init
        left_final = swing_final

    for k in range(N + 1):
        if k <= nb_init_dsp_steps:
            # 초기 DSP: 둘 다 초기 위치 그대로
            left_x[k] = left_init
            right_x[k] = right_init
        elif k <= nb_init_dsp_steps + nb_init_ssp_steps:
            # SSP: 지지발 고정, 스윙발 선형 보간
            alpha = (k - nb_init_dsp_steps) / float(nb_init_ssp_steps)
            swing_pos = (1.0 - alpha) * swing_init + alpha * swing_final
            stance_pos = stance_init
            if stance_is_left:
                left_x[k] = stance_pos
                right_x[k] = swing_pos
            else:
                right_x[k] = stance_pos
                left_x[k] = swing_pos
        else:
            # 마지막 DSP: 두 발 다 최종 위치
            left_x[k] = left_final
            right_x[k] = right_final

    return left_x, right_x, left_final, right_final


def solve_multi_step_mpc(base_params: Parameters, num_steps: int):
    """여러 스텝을 연속으로 풀어서 CoM/발 궤적 전체를 반환."""

    step_len = base_params.end_pos - base_params.start_pos
    N_per = base_params.nb_timesteps

    X_all = None
    left_all = None
    right_all = None

    # 초기 foot step
    left_pos = 0.0
    right_pos = 0.0
    x_prev = np.array([0.0, 0.0, 0.0])

    stance_is_left = True  # 첫 스텝은 왼발 지지로 시작

    for s in range(num_steps):
        # 이번 스텝에서 ZMP current/next 위치
        stance_pos = left_pos if stance_is_left else right_pos
        start_pos = stance_pos
        end_pos = start_pos + step_len

        params_s = Parameters(
            com_height=base_params.com_height,
            dsp_duration=base_params.dsp_duration,
            end_pos=end_pos,
            foot_length=base_params.foot_length,
            gravity=base_params.gravity,
            horizon_duration=base_params.horizon_duration,
            nb_timesteps=base_params.nb_timesteps,
            ssp_duration=base_params.ssp_duration,
            start_pos=start_pos,
        )

        # 1) solve com mpc
        mpc_problem = build_mpc_problem(params_s, initial_state=x_prev)
        plan_s = solve_mpc(mpc_problem, solver="proxqp")
        X_s = plan_s.states  # (N_per+1, 3)

        # 2) foot trajectory
        left_s, right_s, left_pos_end, right_pos_end = build_step_feet_trajectories(
            params_s, N_per, left_pos, right_pos, stance_is_left
        )

        # 3) 전체 궤적에 이어붙이기 (첫 상태 중복 제거)
        if X_all is None:
            X_all = X_s
            left_all = left_s
            right_all = right_s
        else:
            X_all = np.vstack((X_all, X_s[1:, :]))
            left_all = np.concatenate((left_all, left_s[1:]))
            right_all = np.concatenate((right_all, right_s[1:]))

        # 다음 스텝 준비
        x_prev = X_s[-1, :]
        left_pos = left_pos_end
        right_pos = right_pos_end
        stance_is_left = not stance_is_left  # 지지발 교대

    return X_all, left_all, right_all


def animate_humanoid_multi_step(params: Parameters, X_all, left_all, right_all):
    
    N_total = X_all.shape[0] - 1
    com_x = X_all[:, 0]
    hip_z = params.com_height

    xmin = min(com_x.min(), left_all.min(), right_all.min()) - 0.2
    xmax = max(com_x.max(), left_all.max(), right_all.max()) + 0.2

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.1, hip_z + 0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Humanoid MPC: multi-step with alternating legs")

    # ground
    ax.plot([xmin - 0.2, xmax + 0.2], [0.0, 0.0], "k-", linewidth=4)

    # 왼다리(파랑), 오른다리(빨강), 엉덩이(검정)
    left_leg, = ax.plot([], [], "b-", linewidth=4, label="left leg")
    right_leg, = ax.plot([], [], "r-", linewidth=4, label="right leg")
    hip_point, = ax.plot([], [], "ko", markersize=6)

    # 발 위치 점
    left_foot_pt, = ax.plot([], [], "bs", markersize=4)
    right_foot_pt, = ax.plot([], [], "rs", markersize=4)

    ax.legend(loc="upper left")

    def init():
        left_leg.set_data([], [])
        right_leg.set_data([], [])
        hip_point.set_data([], [])
        left_foot_pt.set_data([], [])
        right_foot_pt.set_data([], [])
        return left_leg, right_leg, hip_point, left_foot_pt, right_foot_pt

    def update(frame):
        hx = com_x[frame]
        hz = hip_z
        xl = left_all[frame]
        xr = right_all[frame]

        # 왼/오른 다리
        left_leg.set_data([hx, xl], [hz, 0.0])
        right_leg.set_data([hx, xr], [hz, 0.0])

        # 엉덩이, 발 점
        hip_point.set_data([hx], [hz])
        left_foot_pt.set_data([xl], [0.0])
        right_foot_pt.set_data([xr], [0.0])

        return left_leg, right_leg, hip_point, left_foot_pt, right_foot_pt

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=N_total + 1,
        interval=100,
        blit=True,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    params = Parameters()
    NUM_STEPS = 5  # <-- 여기 숫자 바꾸면 걸음 수가 바뀜

    X_all, left_all, right_all = solve_multi_step_mpc(params, NUM_STEPS)
    animate_humanoid_multi_step(params, X_all, left_all, right_all)
