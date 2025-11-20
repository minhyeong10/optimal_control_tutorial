import logging
from typing import Optional

import numpy as np
import qpsolvers

from .mpc_problem import MPCProblem


class Plan:
    
    __inputs: Optional[np.ndarray]
    __states: Optional[np.ndarray]
    problem: MPCProblem
    qpsol: qpsolvers.Solution

    def __init__(self, problem: MPCProblem, qpsol: qpsolvers.Solution) -> None:

        inputs: Optional[np.ndarray] = None

        if qpsol.found and qpsol.x is not None:
            # qpsol.x: 스택된 입력 벡터 U (shape: m*N,)
            # → (N, m) 형태로 재구성
            U = qpsol.x.reshape((problem.nb_timesteps, problem.input_dim))
            inputs = U

        self.__inputs = inputs       # (N, m) 또는 None
        self.__states = None         # 처음에는 lazy 계산
        self.problem = problem
        self.qpsol = qpsol

    # ------------------------------------------------------------------ #
    # Plan 상태 확인 / 접근자
    # ------------------------------------------------------------------ #
    @property
    def is_empty(self) -> bool:
        
        return self.__inputs is None

    @property
    def first_input(self) -> Optional[np.ndarray]:

        # 첫 번째 제어 입력 u_0 반환
        if self.__inputs is None:
            return None
        return self.__inputs[0]

    @property
    def inputs(self) -> Optional[np.ndarray]:
        
        # 전체 입력 시퀀스 U 반환
        return self.__inputs

    @property
    def states(self) -> Optional[np.ndarray]:
        
        # 상태 궤적 X 반환
        
        # 입력이 없으면 상태 궤적도 만들 수 없다.
        if self.__inputs is None:
            return None

        # 아직 상태를 계산하지 않았다면, 한 번만 계산해 캐싱
        if self.__states is None:
            x_init = self.problem.initial_state
            if x_init is None:
                logging.warning("Problem has undefined initial state.")
                return None

            # integrate: (N+1, state_dim) 궤적 생성
            self.__states = self.problem.integrate(x_init, self.__inputs)

        return self.__states
