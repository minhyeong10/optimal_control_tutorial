import numpy as np
from scipy.linalg import solve_continuous_are

def continuous_lqr(A, B, Q, R):
    """
    연속시간 LQR: xdot = A x + B u
    cost = ∫ (xᵀ Q x + uᵀ R u) dt

    Riccati 방정식 AᵀP + P A - P B R⁻¹ Bᵀ P + Q = 0
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    Q = np.array(Q, dtype=float)
    R = np.array(R, dtype=float)

    # 1) CARE 풀기 → P 구함
    P = solve_continuous_are(A, B, Q, R)

    # 2) 최적 게인 K = R⁻¹ Bᵀ P
    R_inv = np.linalg.inv(R)
    K = R_inv @ B.T @ P

    return K, P
