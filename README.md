# 🧠 Optimal Control Tutorial

Python implementations of fundamental **Optimal Control** algorithms:

- **LQR** – Linear Quadratic Regulator  
- **iLQR / DDP** – Iterative LQR / Differential Dynamic Programming  
- **MPC** – Model Predictive Control  

This repository is organized for **learning / teaching optimal control**, with clean and minimal Python implementations.

---

## 📂 Repository Structure
```text
Optimal_Control/
│
├── LQR/
│ ├── lqr.py # Continuous-time LQR (solve CARE)
│ ├── double_integrator_lqr.py # LQR demo on double integrator
│ └── manipulator_lqr.py # LQR for n-DOF manipulator (via linearization)
│
├── iLQR/
│ ├── iLQR.py # iLQR / iLQG solver
│ ├── boxQP.py # Box-constrained QP solver for control limits
│ └── demo_inverted_pendulum.py# iLQR demo: inverted pendulum
│
├── MPC/
│ ├── qpmpc/
│ │ ├── mpc_problem.py # Define linear MPC problem
│ │ ├── mpc_qp.py # Convert MPC → QP
│ │ ├── plan.py # Container for MPC results
│ │ └── solve_mpc.py # Solve MPC using qpsolvers
│ └── examples/ (TODO)
│
└── viz/
├── LQR_manipulator.gif
├── iLQR_inverted_pendulum.gif
├── bipedal_mpc_onestep.gif
└── bipedal_mpc_multistep.gif

```

---

🎯 Algorithms Overview (No Equations)
1️⃣ LQR — Linear Quadratic Regulator

LQR is an optimal control method for linear systems.
It computes a state-feedback controller that minimizes a quadratic cost on states and control inputs.

Key ideas:

Assumes linear dynamics

Penalizes deviation from desired state

Produces an optimal control law of the form u = -Kx

Very fast and widely used in robotics & control

✔️ LQR Manipulator Example
<img src="viz/LQR_manipulator.gif" width="400">
2️⃣ iLQR / DDP — Iterative LQR

iLQR generalizes LQR to nonlinear systems.

Main procedure:

Linearize nonlinear dynamics locally

Quadratically approximate the cost

Perform an LQR backward pass to compute gains

Apply line-search updates to refine the solution

Repeat until convergence

This makes iLQR suitable for pendulums, manipulators, and complex nonlinear robots.

✔️ iLQR Inverted Pendulum Demo
<img src="viz/iLQR_inverted_pendulum.gif" width="400">
3️⃣ MPC — Model Predictive Control

MPC solves a finite-horizon optimal control problem at every timestep.

Characteristics:

Predicts future states over a horizon

Optimizes control inputs while respecting constraints

Applies only the first control input

Repeats the process at the next timestep

Great for robots that must follow trajectories or stay within limits

This repository converts MPC into a Quadratic Program (QP) and solves it using QP solvers.

✔️ MPC Bipedal Example (one-step)
<img src="viz/bipedal_mpc_onestep.gif" width="400">
✔️ MPC Bipedal Example (multi-step)
<img src="viz/bipedal_mpc_multistep.gif" width="400">

---

# ⚙️ Requirements

Install dependencies:

```bash
pip install numpy scipy matplotlib qpsolvers
    
