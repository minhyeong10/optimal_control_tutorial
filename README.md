🧠 Optimal Control Tutorial

Python implementations of fundamental Optimal Control algorithms:

LQR – Linear Quadratic Regulator

iLQR / DDP – Iterative LQR / Differential Dynamic Programming

MPC – Model Predictive Control

This repository is organized for learning / teaching optimal control, with clean and minimal Python implementations.

📂 Repository Structure
Optimal_Control/
│
├── LQR/
│   ├── lqr.py                    # Continuous-time LQR (solve CARE)
│   ├── double_integrator_lqr.py # LQR demo on double integrator
│   └── manipulator_lqr.py       # LQR for n-DOF manipulator (via linearization)
│
├── iLQR/
│   ├── iLQR.py                   # iLQR / iLQG solver
│   ├── boxQP.py                  # Box-constrained QP solver for control limits
│   └── demo_inverted_pendulum.py# iLQR demo: inverted pendulum
│
├── MPC/
│   ├── qpmpc/
│   │   ├── mpc_problem.py        # Define linear MPC problem
│   │   ├── mpc_qp.py             # Convert MPC → QP
│   │   ├── plan.py               # Container for MPC results
│   │   └── solve_mpc.py          # Solve MPC using qpsolvers
│   └── examples/ (TODO)
│
└── viz/
    ├── LQR_manipulator.gif
    ├── iLQR_inverted_pendulum.gif
    ├── bipedal_mpc_onestep.gif
    └── bipedal_mpc_multistep.gif
