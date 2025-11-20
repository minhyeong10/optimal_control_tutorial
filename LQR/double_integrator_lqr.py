#!/usr/bin/env python3
import numpy as np

from solvers.lqr import continuous_lqr

def main():
    # Double integrator:
    #   x = [q, qdot]
    #   qddot = u  -> xdot = A x + B u
    A = np.array([[0., 1.],
                  [0., 0.]])
    B = np.array([[0.],
                  [1.]])

    Q = np.eye(2)   # 상태 가중치
    R = np.eye(1)   # 입력 가중치

    K, S = continuous_lqr(A, B, Q, R)

    print("=== Double Integrator LQR (manual, from solvers.lqr) ===")
    print("A =\n", A)
    print("B =\n", B)
    print("Q =\n", Q)
    print("R =\n", R)
    print()
    print("K (state feedback gain) = ", K)
    print("S (solution of Riccati) =\n", S)

if __name__ == "__main__":
    main()
