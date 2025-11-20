import logging
from typing import Optional

import numpy as np
import qpsolvers
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix

from .mpc_problem import MPCProblem

class MPCQP:
    
    G: np.ndarray
    P: np.ndarray
    Phi: np.ndarray
    Psi: np.ndarray
    h: np.ndarray
    phi_last: np.ndarray
    psi_last: np.ndarray
    q: np.ndarray
    e: np.ndarray
    C: Optional[np.ndarray]

    def __init__(self, mpc_problem: MPCProblem, sparse: bool = False) -> None:
        
        # 차원 설정
        input_dim: int = mpc_problem.input_dim
        state_dim: int = mpc_problem.state_dim
        nb_steps: int = mpc_problem.nb_timesteps

        stacked_input_dim = input_dim * nb_steps  # m * N

        # 초기 상태 필요
        x_init: np.ndarray = mpc_problem.initial_state

        # -----------------------------
        # 상태 예측: x_k = phi_k x0 + psi_k U
        # -----------------------------
        # 루프 invariant:
        #   x_k = phi @ x_init + psi @ U
        phi = np.eye(state_dim)
        psi = np.zeros((state_dim, stacked_input_dim))

        G_blocks = []
        h_blocks = []
        phi_blocks = []
        psi_blocks = []
        e_blocks = []
        C_blocks = []

        for k in range(nb_steps):
            # 현재 시간 k에서의 예측 행렬 저장
            phi_blocks.append(phi)
            psi_blocks.append(psi)

            # 시스템 / 제약 행렬
            A_k = mpc_problem.get_transition_state_matrix(k)
            B_k = mpc_problem.get_transition_input_matrix(k)
            C_k = mpc_problem.get_ineq_state_matrix(k)   # G_x
            D_k = mpc_problem.get_ineq_input_matrix(k)   # G_u
            e_k = mpc_problem.get_ineq_vector(k)         # h_k

            # 이 시간 step에서 전체 U에 대한 inequality 행렬 G_k
            #   C_k (x_k) + D_k (u_k) ≤ e_k
            #   x_k = phi x_init + psi U → G_k U ≤ h_k 로 바꿈
            G_k = np.zeros((e_k.shape[0], stacked_input_dim), dtype=float)

            if C_k is None:
                # C_k x_k term 없음 → 상수항 그대로 e_k
                h_k = e_k
            else:
                # C_k (phi x_init) 쪽은 상수로 들어가므로 우변으로 넘김
                # e_k - C_k phi x_init
                h_k = e_k - C_k.dot(phi).dot(x_init)  # type: ignore

            # 현재 u_k가 U 전체에서 차지하는 슬라이스
            inp_slice = slice(k * input_dim, (k + 1) * input_dim)

            # 입력 제약 부분: D_k u_k
            if D_k is not None:
                # 초기값이 0이므로 += 대신 대입 가능
                G_k[:, inp_slice] = D_k

            # 상태 제약 부분: C_k psi U
            if C_k is not None:
                G_k += C_k.dot(psi)  # type: ignore

            # k=0, 입력 제약 없고, 초기 상태로 이미 h_k<0이면 완전 infeasible
            if k == 0 and D_k is None and np.any(h_k < 0.0):
                logging.warning(
                    "initial state is infeasible: "
                    f"G_0 * U <= h_0 with G_0 == 0 and min(h_0) == {np.min(h_k)}"
                )

            G_blocks.append(G_k)
            h_blocks.append(h_k)
            e_blocks.append(e_k)
            C_blocks.append(C_k)

            # 다음 step을 위한 phi, psi 업데이트
            # x_{k+1} = A_k x_k + B_k u_k
            #        = A_k (phi x_init + psi U) + B_k u_k
            #        = (A_k phi) x_init + (A_k psi + B_k E_k) U
            phi = A_k.dot(phi)
            psi = A_k.dot(psi)
            psi[:, inp_slice] = B_k  # 이 step의 u_k가 들어가는 위치

        # -----------------------------
        # 큰 블록 행렬/벡터로 스택
        # -----------------------------
        G = np.vstack(G_blocks).astype(float)    # (n_ineq_total, mN)
        h = np.hstack(h_blocks).astype(float)    # (n_ineq_total,)
        Phi = np.vstack(phi_blocks).astype(float)  # (N * n, n)
        Psi = np.vstack(psi_blocks).astype(float)  # (N * n, mN)
        e = np.hstack(e_blocks).astype(float)      # (n_ineq_total,)

        # C는 block-diagonal 형태 (각 step의 C_k가 block으로 쌓임)
        # C_k가 모두 None인 경우에는 C=None
        C = block_diag(*C_blocks) if C_blocks else None

        # -----------------------------
        # Quadratic term P, linear term q 초기값
        # cost = ½ Uᵀ P U + qᵀ U
        # -----------------------------
        P = (
            mpc_problem.stage_input_cost_weight
            * np.eye(stacked_input_dim, dtype=float)
        )

        # terminal cost: || x_N - x_goal ||² = || phi_last x0 + psi_last U - x_goal ||²
        # → Uᵀ (psi_lastᵀ psi_last) U + ...
        if mpc_problem.terminal_cost_weight is not None:
            P += (
                mpc_problem.terminal_cost_weight
                * psi.T.dot(psi)
            )

        # stage state cost: Σ_k || x_k - x_ref_k ||²
        # x_k = Phi_k x0 + Psi_k U → 전체를 스택하면 Psi 사용
        if mpc_problem.stage_state_cost_weight is not None:
            P += (
                mpc_problem.stage_state_cost_weight
                * Psi.T.dot(Psi)
            )

        q = np.zeros(stacked_input_dim, dtype=float)

        # -----------------------------
        # 멤버 저장
        # -----------------------------
        self.G = csc_matrix(G) if sparse else G
        self.P = csc_matrix(P) if sparse else P

        self.Phi = Phi          # 전체 예측 phi 블록
        self.Psi = Psi          # 전체 예측 psi 블록
        self.h = h              # inequality RHS
        self.phi_last = phi     # 마지막 step의 phi_N
        self.psi_last = psi     # 마지막 step의 psi_N
        self.e = e              # 원래 e 블록 스택
        self.C = C              # block diag(C_k)
        self.q = q              # 아래 update_cost_vector에서 실제 값 설정

        # 초기 상태/타깃 기준으로 q 업데이트
        self.update_cost_vector(mpc_problem)

    # ------------------------------------------------------------------ #
    # QP 인터페이스
    # ------------------------------------------------------------------ #
    @property
    def problem(self) -> qpsolvers.Problem:
        """qpsolvers에 넘길 수 있는 QP 문제 객체."""
        return qpsolvers.Problem(self.P, self.q, self.G, self.h)

    # ------------------------------------------------------------------ #
    # cost vector 업데이트 (x0, goal, target_states 바뀔 때)
    # ------------------------------------------------------------------ #
    def update_cost_vector(self, mpc_problem: MPCProblem) -> None:
        
        x_init = mpc_problem.initial_state

        # q 초기화
        self.q[:] = 0.0

        # terminal cost: || x_N - x_goal ||²
        if mpc_problem.has_terminal_cost:
            # x_N = phi_last x_init + psi_last U
            # → cost ~ || x_N - x_goal ||²
            #   = Uᵀ (psiᵀ psi) U + 2 (phi x_init - x_goal)ᵀ psi U + const
            # → q_term = (2 * ...)ᵀ / 2 = (...)ᵀ
            c_N = self.phi_last.dot(x_init) - mpc_problem.goal_state
            self.q += (
                mpc_problem.terminal_cost_weight
                * c_N.T.dot(self.psi_last)
            )

        # stage state cost: Σ_k || x_k - x_ref_k ||²
        if mpc_problem.has_stage_state_cost:
            # x_stack = Phi x_init + Psi U
            # x_ref_stack = target_states
            c_stack = self.Phi.dot(x_init) - mpc_problem.target_states
            self.q += (
                mpc_problem.stage_state_cost_weight
                * c_stack.T.dot(self.Psi)
            )

    # ------------------------------------------------------------------ #
    # inequality RHS 업데이트 (x0만 바뀌었을 때)
    # ------------------------------------------------------------------ #
    def update_constraint_vector(self, mpc_problem: MPCProblem) -> None:

        if self.C is not None:
            # 원래 제약: C x ≤ e, x = Phi x_init + Psi U
            # → C Phi x_init + C Psi U ≤ e
            # → G U ≤ h,  h = e - C Phi x_init
            h = self.e - self.C @ self.Phi @ mpc_problem.initial_state
            self.h = h.flatten()
