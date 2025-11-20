from qpsolvers import solve_problem

from .mpc_problem import MPCProblem
from .mpc_qp import MPCQP
from .plan import Plan


def solve_mpc(
    problem: MPCProblem,
    solver: str,
    sparse: bool = False,
    **kwargs,
) -> Plan:
    
    mpc_qp = MPCQP(problem, sparse=sparse)
    qpsol = solve_problem(mpc_qp.problem, solver=solver, **kwargs)
    return Plan(problem, qpsol)
