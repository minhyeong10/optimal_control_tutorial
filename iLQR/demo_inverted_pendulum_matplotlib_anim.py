import numpy as np
from iLQR import iLQR
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class InvertedPendulumEnv:
    """
    Inverted pendulum environment with:

      x = [sin(theta), cos(theta), omega]
      u = torque

    Dynamics are discretized with dt, and cost is quadratic around
    the upright equilibrium x = [0, 1, 0], u = 0.
    """

    def __init__(self):
        # -----------------------------
        # Environment parameters
        # -----------------------------
        self.m = 1.0
        self.l = 1.0
        self.g = 10.0
        self.dt = 0.05
        self.T = 200  # horizon length

        # -----------------------------
        # Cost function parameters
        # -----------------------------
        self.Q1 = 10.0   # weight on angle error via sin/cos
        self.Q2 = 0.1    # weight on angular velocity
        self.R = 0.001   # weight on control

        # -----------------------------
        # State / action dimensions
        # -----------------------------
        self.x = None
        self.u = None
        self.x_dim = 3
        self.u_dim = 1

        # control limits [umin, umax] for each dimension
        self.u_lims = np.array([[-2.0, 2.0]])

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def dynamics(self, x, u):
        """
        One-step dynamics.

        Parameters
        ----------
        x : np.ndarray, shape (3, K)
            x[0] = sin(theta), x[1] = cos(theta), x[2] = omega.
        u : np.ndarray, shape (1, K)
            torque.

        Returns
        -------
        x_new : np.ndarray, shape (3, K)
        """
        # x = [sinθ, cosθ, ω]
        theta_dot_dt = x[2] * self.dt
        c = np.cos(theta_dot_dt)
        s = np.sin(theta_dot_dt)

        x_new0 = x[0] * c + x[1] * s
        x_new1 = x[1] * c - x[0] * s
        x_new2 = x[2] + (3 * self.g / (2 * self.l)) * x[0] * self.dt \
                       + (3 / (self.m * self.l ** 2)) * u[0] * self.dt

        x_new = np.array([x_new0, x_new1, x_new2])
        return x_new

    def dynamics_dx(self, x, u):
        """
        Jacobian df/dx along a trajectory.

        Parameters
        ----------
        x : (3, N)
        u : (1, N)

        Returns
        -------
        dfdx : (N, 3, 3)
        """
        n = x.shape[0]
        N = x.shape[1]
        dfdx = np.zeros((N, n, n))

        for i in range(N):
            theta_dot_dt = x[2, i] * self.dt
            c = np.cos(theta_dot_dt)
            s = np.sin(theta_dot_dt)

            dfdx[i, :, :] = np.array([
                [c,               s, (-x[0, i] * s + x[1, i] * c) * self.dt],
                [-s,              c, (-x[0, i] * c - x[1, i] * s) * self.dt],
                [3 * self.g * self.dt / (2 * self.l), 0.0, 1.0]
            ])

        return dfdx

    def dynamics_du(self, x, u):
        """
        Jacobian df/du along a trajectory.

        Parameters
        ----------
        x : (3, N)
        u : (1, N)

        Returns
        -------
        dfdu : (N, 3, 1)
        """
        n = x.shape[0]
        m = u.shape[0]
        N = x.shape[1]
        dfdu = np.zeros((N, n, m))

        for i in range(N):
            dfdu[i, :, :] = np.array([
                [0.0],
                [0.0],
                [3 * self.dt / (self.m * self.l ** 2)]
            ])

        return dfdu

    # ------------------------------------------------------------------ #
    # Cost
    # ------------------------------------------------------------------ #
    def cost(self, x, u):
        """
        Stage cost l(x,u).

        Parameters
        ----------
        x : (3, K)
        u : (1, K)

        Returns
        -------
        l : (K,) or scalar
        """
        # x[0]=sinθ, x[1]=cosθ → θ=0 이 목표 (x=[0,1,0])
        l = (x[0] ** 2) * self.Q1 \
          + ((x[1] - 1.0) ** 2) * self.Q1 \
          + (x[2] ** 2) * self.Q2 \
          + (u[0] ** 2) * self.R
        return l

    def cost_dx(self, x, u):
        """
        Gradient dl/dx.

        Parameters
        ----------
        x : (3, K) or (3,)
        u : (1, K) or (1,)

        Returns
        -------
        dldx : (3, K) or (3,)
        """
        dldx = np.array([
            2.0 * x[0] * self.Q1,
            2.0 * (x[1] - 1.0) * self.Q1,
            2.0 * x[2] * self.Q2
        ])
        return dldx

    def cost_du(self, x, u):
        """
        Gradient dl/du.

        Parameters
        ----------
        x : (3, K) or (3,)
        u : (1, K) or (1,)

        Returns
        -------
        dldu : (1, K) or (1,)
        """
        dldu = np.array([
            2.0 * u[0] * self.R
        ])
        return dldu

    def cost_dxx(self, x, u):
        """
        Hessian d^2 l / dx^2 along trajectory.

        Parameters
        ----------
        x : (3, N)
        u : (1, N)

        Returns
        -------
        dldxx : (N, 3, 3)
        """
        n = x.shape[0]
        N = x.shape[1]
        dldxx = np.zeros((N, n, n))

        H = np.array([
            [2.0 * self.Q1, 0.0,            0.0],
            [0.0,           2.0 * self.Q1,  0.0],
            [0.0,           0.0,            2.0 * self.Q2]
        ])
        for i in range(N):
            dldxx[i, :, :] = H

        return dldxx

    def cost_duu(self, x, u):
        """
        Hessian d^2 l / du^2 along trajectory.

        Parameters
        ----------
        x : (3, N)
        u : (1, N)

        Returns
        -------
        dlduu : (N, 1, 1)
        """
        m = u.shape[0]
        N = u.shape[1]
        dlduu = np.zeros((N, m, m))

        H = np.array([[2.0 * self.R]])
        for i in range(N):
            dlduu[i, :, :] = H

        return dlduu

    def cost_dux(self, x, u):
        """
        Cross term d^2 l / du dx along trajectory.

        Parameters
        ----------
        x : (3, N)
        u : (1, N)

        Returns
        -------
        dldux : (N, 1, 3)
        """
        n = x.shape[0]
        m = u.shape[0]
        N = x.shape[1]
        dldux = np.zeros((N, m, n))  # no cross term
        return dldux

    # ------------------------------------------------------------------ #
    # Dynamics + Cost wrapper (for iLQR)
    # ------------------------------------------------------------------ #
    def dyn_cst(self, x, u):
        """
        Combined dynamics & cost function for iLQR.

        Parameters
        ----------
        x : (3, T) or (3, K)
        u : (1, T) or (1, K)

        Returns
        -------
        f   : next state(s)
        c   : cost(s)
        fx  : df/dx
        fu  : df/du
        cx  : dl/dx
        cu  : dl/du
        cxx : d^2l/dx^2
        cuu : d^2l/du^2
        cux : d^2l/dudx
        """
        f = self.dynamics(x, u)
        c = self.cost(x, u)
        fx = self.dynamics_dx(x, u)
        fu = self.dynamics_du(x, u)
        cx = self.cost_dx(x, u)
        cu = self.cost_du(x, u)
        cxx = self.cost_dxx(x, u)
        cuu = self.cost_duu(x, u)
        cux = self.cost_dux(x, u)
        return f, c, fx, fu, cx, cu, cxx, cuu, cux

    # ------------------------------------------------------------------ #
    # Misc utilities
    # ------------------------------------------------------------------ #
    def reset(self):
        """
        Random initial state: theta ~ U(-pi, pi), omega ~ U(-1,1)

        Returns
        -------
        x0 : (3,1)
        """
        # [theta, omega] sampling
        high = np.array([[np.pi], [1.0]])
        x_reduced = np.random.uniform(low=-high, high=high)
        self.x = self.augment_state(x_reduced)
        return self.x

    @staticmethod
    def augment_state(x_reduced):
        """
        Convert [theta, omega] to [sin(theta), cos(theta), omega].

        Parameters
        ----------
        x_reduced : (2,1) or (2,)

        Returns
        -------
        x : (3,1)
        """
        theta = x_reduced[0]
        omega = x_reduced[1]
        return np.array([np.sin(theta), np.cos(theta), omega]).reshape(3, 1)

    @staticmethod
    def reduce_state(x):
        """
        Convert [sin(theta), cos(theta), omega] to [theta, omega].

        Parameters
        ----------
        x : (3,1) or (3,)

        Returns
        -------
        x_reduced : (2,)
        """
        theta = np.arctan2(x[0], x[1])
        omega = x[2]
        return np.array([theta, omega])

    @staticmethod
    def angle_normalize(theta):
        """
        Wrap angle to [-pi, pi].
        """
        return ((theta + np.pi) % (2 * np.pi)) - np.pi


# ====================================================================== #
# Main: iLQR on inverted pendulum
# ====================================================================== #
if __name__ == "__main__":
    env = InvertedPendulumEnv()

    # -----------------------------
    # Initial state
    # -----------------------------
    x0 = env.reset()               # (3,1)
    x0_reduced = env.reduce_state(x0)  # [theta, omega]

    # -----------------------------
    # Initial control sequence
    # -----------------------------
    u_high = np.tile(np.array([[0.01]]), env.T)  # very small random torque range
    u0 = np.random.uniform(low=-u_high, high=u_high)  # (1, T)

    DYNCST = lambda x, u: env.dyn_cst(x, u)

    # -----------------------------
    # Run iLQR
    # -----------------------------
    x_trj, u_trj, cost = iLQR(DYNCST, x0, u0, env.u_lims)
    # x_trj: (3, N+1), u_trj: (1, N)

    N = u_trj.shape[1]
    dt = env.dt
    t = np.arange(N + 1) * dt

    sin_th = x_trj[0, :]
    cos_th = x_trj[1, :]
    theta = np.arctan2(sin_th, cos_th)  # [-pi, pi]
    omega = x_trj[2, :]
    u = u_trj[0, :]

    # ================================================================== #
    # (1) Plot state and input trajectories
    # ================================================================== #
    fig1, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # theta
    axs[0].plot(t, theta, label=r"$\theta$ (rad)")
    axs[0].axhline(0.0, color="k", linestyle="--", linewidth=0.5)
    axs[0].set_ylabel("theta [rad]")
    axs[0].legend()
    axs[0].grid(True)

    # omega
    axs[1].plot(t, omega, label=r"$\dot{\theta}$ (rad/s)")
    axs[1].axhline(0.0, color="k", linestyle="--", linewidth=0.5)
    axs[1].set_ylabel("omega [rad/s]")
    axs[1].legend()
    axs[1].grid(True)

    # control
    axs[2].step(t[:-1], u, where="post", label="u (torque)")
    axs[2].axhline(env.u_lims[0, 0], color="r", linestyle="--",
                   linewidth=0.5, label="u min/max")
    axs[2].axhline(env.u_lims[0, 1], color="r", linestyle="--",
                   linewidth=0.5)
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("u")
    axs[2].legend()
    axs[2].grid(True)

    fig1.tight_layout()

    # ================================================================== #
    # (2) Animation of the pendulum
    # ================================================================== #
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    L = env.l

    ax2.set_xlim(-1.2 * L, 1.2 * L)
    ax2.set_ylim(-1.2 * L, 1.2 * L)
    ax2.set_aspect("equal", "box")
    ax2.grid(True)
    ax2.set_title("Inverted Pendulum (iLQR trajectory)")

    # rod + point mass
    line, = ax2.plot([], [], "o-", lw=3)

    # time text
    time_text = ax2.text(0.05, 0.9, "", transform=ax2.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return line, time_text

    def update(k):
        # theta=0 -> pointing up: x = L*sin(theta), y = L*cos(theta)
        x_tip = L * np.sin(theta[k])
        y_tip = L * np.cos(theta[k])

        line.set_data([0.0, x_tip], [0.0, y_tip])
        time_text.set_text(f"t = {t[k]:.2f} s")
        return line, time_text

    ani = FuncAnimation(
        fig2,
        update,
        frames=len(t),
        init_func=init,
        blit=True,
        interval=dt * 1000,  # ms
    )

    print("Initial State (reduced): theta = %.3f pi, omega = %.3f"
          % (x0_reduced[0] / np.pi, x0_reduced[1]))
    print("Final cost: ", np.sum(cost))

    plt.show()
