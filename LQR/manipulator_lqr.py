#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 등록용)
from scipy.linalg import solve_continuous_are

# ======================================
# LQR
# ======================================
def continuous_lqr(A, B, Q, R):
    
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    Q = np.array(Q, dtype=float)
    R = np.array(R, dtype=float)

    P = solve_continuous_are(A, B, Q, R)
    R_inv = np.linalg.inv(R)
    K = R_inv @ B.T @ P
    return K, P

# ======================================
# 3-자유도 매니퓰레이터
# ======================================
N_JOINTS = 3
LINK_LENGTHS = np.array([0.7, 0.6, 0.4])
JOINT_AXES = ["z", "y", "y"]

# ======================================
# Ref trajectory
# ======================================
def joint_reference(t: float):
    
    # joint1: 1.2 * sin(0.5 t)
    q1 = 1.2 * np.sin(0.5 * t)
    q1d = 1.2 * 0.5 * np.cos(0.5 * t)
    q1dd = -1.2 * (0.5 ** 2) * np.sin(0.5 * t)
    
    # joint2: -0.5 + 1.0 * sin(0.8 t)
    q2 = -0.5 + 1.0 * np.sin(0.8 * t)
    q2d = 1.0 * 0.8 * np.cos(0.8 * t)
    q2dd = -1.0 * (0.8 ** 2) * np.sin(0.8 * t)

    # joint3: 1.5 * sin(1.2 t)
    q3 = 1.5 * np.sin(1.2 * t)
    q3d = 1.5 * 1.2 * np.cos(1.2 * t)
    q3dd = -1.5 * (1.2 ** 2) * np.sin(1.2 * t)

    q_ref = np.array([q1, q2, q3])
    qd_ref = np.array([q1d, q2d, q3d])
    qdd_ref = np.array([q1dd, q2dd, q3dd])

    return q_ref, qd_ref, qdd_ref

# ======================================
# Forward Kinematics
# ======================================
def rot_matrix(axis, angle):
    
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == "z":
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]])
    elif axis == "y":
        return np.array([[ c, 0.0,  s],
                         [0.0, 1.0, 0.0],
                         [-s, 0.0,  c]])
    else:
        raise ValueError("axis must be 'z' or 'y'")


def fk_3d(q):
    
    xs = [0.0]
    ys = [0.0]
    zs = [0.0]

    R = np.eye(3)
    p = np.zeros(3)

    for i in range(N_JOINTS):
        R = R @ rot_matrix(JOINT_AXES[i], q[i])  # 누적 회전
        dp = R @ np.array([LINK_LENGTHS[i], 0.0, 0.0])
        p = p + dp
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])

    return np.array(xs), np.array(ys), np.array(zs)

# ======================================
# tracking용 동역학 모델 -> 간단하게 double integrator
# ======================================
def build_double_integrator_system(nq: int):
    
    n = nq
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [np.zeros((n, n)), np.zeros((n, n))]
    ])
    B = np.block([
        [np.zeros((n, n))],
        [np.eye(n)]
    ])
    return A, B

# ======================================
# tracking 시뮬레이션
# ======================================
def simulate_lqr_tracking(T_total=8.0, dt=0.005):
    
    nq = N_JOINTS
    nx = 2 * nq

    A, B = build_double_integrator_system(nq)
    Q = np.diag(
        np.concatenate([
            50.0 * np.ones(nq),  # q error
            5.0 * np.ones(nq),   # qdot error
        ])
    )
    R = 0.1 * np.eye(nq)

    K, P = continuous_lqr(A, B, Q, R)
    print("LQR K =\n", K)



    times = np.arange(0.0, T_total, dt)
    N = len(times)

    # 상태 초기값: 정지 상태에서 시작
    x = np.zeros(nx)  # [q; qdot]

    X_trj = np.zeros((N, nx))
    q_ref_trj = np.zeros((N, nq))

    def f(x, t):
        # 참조
        q_ref, qd_ref, qdd_ref = joint_reference(t)
        x_ref = np.concatenate([q_ref, qd_ref])  # (6,)
        u_ff = qdd_ref                           # (3,)

        e = x - x_ref
        u = u_ff - K @ e 

        xdot = A @ x + B @ u
        return xdot

    # Smoothing
    for i, t in enumerate(times):
        X_trj[i, :] = x
        q_ref, _, _ = joint_reference(t)
        q_ref_trj[i, :] = q_ref

        k1 = f(x, t)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    q_trj = X_trj[:, :nq]   # 실제 조인트
    qd_trj = X_trj[:, nq:]  # 속도 (원하면 쓸 수 있음)

    return times, q_trj, q_ref_trj

# ======================================
# 3D 애니메이션
# ======================================
def animate_manipulator_3d(times, q_trj, q_ref_trj=None):
    
    all_x, all_y, all_z = [], [], []
    for i in range(len(times)):
        xs, ys, zs = fk_3d(q_trj[i, :])
        all_x.append(xs)
        all_y.append(ys)
        all_z.append(zs)
    all_x = np.vstack(all_x)
    all_y = np.vstack(all_y)
    all_z = np.vstack(all_z)

    reach = np.sum(LINK_LENGTHS)
    margin = 0.3 * reach

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-reach - margin, reach + margin)
    ax.set_ylim(-reach - margin, reach + margin)
    ax.set_zlim(0.0, reach + margin)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3-DoF Manipulator (LQR tracking, 3D animation)")
    ax.view_init(elev=30, azim=40)  # 카메라 각도

    line, = ax.plot([], [], [], "-o", lw=4, label="arm")
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    def init():
        line.set_data_3d([], [], [])
        time_text.set_text("")
        return line, time_text

    def update(frame):
        xs = all_x[frame, :]
        ys = all_y[frame, :]
        zs = all_z[frame, :]
        line.set_data_3d(xs, ys, zs)
        time_text.set_text(f"t = {times[frame]:.2f} s")
        return line, time_text

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(times),
        interval=10,   # 10ms → ~100 FPS (부드럽게)
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()

    if q_ref_trj is not None:
        fig2, axs = plt.subplots(N_JOINTS, 1, figsize=(6, 6), sharex=True)
        for j in range(N_JOINTS):
            axs[j].plot(times, q_trj[:, j], label=f"q{j+1} (actual)")
            axs[j].plot(times, q_ref_trj[:, j], "--", label=f"q{j+1}_ref")
            axs[j].grid(True)
            axs[j].legend(loc="upper right")
        axs[-1].set_xlabel("time [s]")
        fig2.suptitle("Joint angle tracking (LQR)")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # LQR tracking으로 joint trajectory 생성
    times, q_trj, q_ref_trj = simulate_lqr_tracking(T_total=8.0, dt=0.01)

    # 3D 애니메이션 + joint tracking plot
    animate_manipulator_3d(times, q_trj, q_ref_trj)
