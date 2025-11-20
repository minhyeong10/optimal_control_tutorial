from typing import List, Optional, Union
import numpy as np

class MPCProblem:
    r"""
    Linear time-varying model predictive control (MPC) problem.

    Discrete-time linear dynamics:

        x_{k+1} = A_k x_k + B_k u_k

    여기서 x는 보통 [p, p_dot] 같이 위치/속도를 스택한 상태벡터라고 가정.

    선형 부등식 제약:

        x_0 = x_init
        C_k x_k + D_k u_k ≤ e_k

    비용 함수는 다음 세 가지 항의 가중 합:

    - Terminal state cost:
        || x_N - x_goal ||^2   (weight = terminal_cost_weight)
    - Stage state tracking cost:
        Σ_k || x_k - x_ref_k ||^2   (weight = stage_state_cost_weight)
    - Stage input cost:
        Σ_k || u_k ||^2             (weight = stage_input_cost_weight)

    Attributes:
        transition_state_matrix   (A_k 또는 A)
        transition_input_matrix   (B_k 또는 B)
        ineq_state_matrix         (C_k 또는 C)
        ineq_input_matrix         (D_k 또는 D)
        ineq_vector               (e_k 또는 e)
        initial_state             (x_init)
        goal_state                (x_goal)
        target_states             (x_ref trajectory)
    """

    # Type
    goal_state: Optional[np.ndarray]
    ineq_input_matrix: Union[None, np.ndarray, List[np.ndarray]]
    ineq_state_matrix: Union[None, np.ndarray, List[np.ndarray]]
    ineq_vector: Union[np.ndarray, List[np.ndarray]]
    initial_state: Optional[np.ndarray]
    input_dim: int
    nb_timesteps: int
    stage_input_cost_weight: float
    stage_state_cost_weight: Optional[float]
    state_dim: int
    target_states: Optional[np.ndarray]
    terminal_cost_weight: Optional[float]
    transition_input_matrix: Union[np.ndarray, List[np.ndarray]]
    transition_state_matrix: Union[np.ndarray, List[np.ndarray]]

    def __init__(
        self,
        transition_state_matrix: Union[np.ndarray, List[np.ndarray]],
        transition_input_matrix: Union[np.ndarray, List[np.ndarray]],
        ineq_state_matrix: Union[None, np.ndarray, List[np.ndarray]],
        ineq_input_matrix: Union[None, np.ndarray, List[np.ndarray]],
        ineq_vector: Union[np.ndarray, List[np.ndarray]],
        nb_timesteps: int,
        terminal_cost_weight: Optional[float],
        stage_state_cost_weight: Optional[float],
        stage_input_cost_weight: float,
        initial_state: Optional[np.ndarray] = None,
        goal_state: Optional[np.ndarray] = None,
        target_states: Optional[np.ndarray] = None,
    ) -> None:
        
        # A_k, B_k 가 time-varying 인지, time-invariant 인지에 따라 차원 추출
        input_dim = (
            transition_input_matrix.shape[1]
            if isinstance(transition_input_matrix, np.ndarray)
            else transition_input_matrix[0].shape[1]
        )
        state_dim = (
            transition_state_matrix.shape[1]
            if isinstance(transition_state_matrix, np.ndarray)
            else transition_state_matrix[0].shape[1]
        )

        # ---------- 속성 저장 ----------
        self.transition_state_matrix = transition_state_matrix
        self.transition_input_matrix = transition_input_matrix

        self.ineq_state_matrix = ineq_state_matrix
        self.ineq_input_matrix = ineq_input_matrix
        self.ineq_vector = ineq_vector

        self.nb_timesteps = nb_timesteps
        self.state_dim = state_dim
        self.input_dim = input_dim

        self.terminal_cost_weight = terminal_cost_weight
        self.stage_state_cost_weight = stage_state_cost_weight
        self.stage_input_cost_weight = stage_input_cost_weight

        # 나중에 setter로 제대로 설정
        self.goal_state = None
        self.initial_state = None
        self.target_states = None

        # ---------- 초기/목표/레퍼런스가 주어졌다면 바로 세팅 ----------
        if goal_state is not None:
            self.update_goal_state(goal_state)
        if initial_state is not None:
            self.update_initial_state(initial_state)
        if target_states is not None:
            self.update_target_states(target_states)

    # ------------------------------------------------------------------ #
    # 비용 항 유무 체크
    # ------------------------------------------------------------------ #
    @property
    def has_terminal_cost(self) -> bool:
        """Terminal state cost가 설정되어 있는지 여부."""
        cost_is_set = (
            self.terminal_cost_weight is not None
            and self.terminal_cost_weight > 1e-10
        )
        return cost_is_set

    @property
    def has_stage_state_cost(self) -> bool:
        """Stage state tracking cost가 설정되어 있는지 여부."""
        cost_is_set = (
            self.stage_state_cost_weight is not None
            and self.stage_state_cost_weight > 1e-10
        )
        return cost_is_set

    # ------------------------------------------------------------------ #
    # A_k, B_k, C_k, D_k, e_k 가져오기 (time-varying / invariant 대응)
    # ------------------------------------------------------------------ #
    def get_transition_state_matrix(self, k: int) -> np.ndarray:
        """k 시점의 상태 전이 행렬 A_k 반환."""
        return (
            self.transition_state_matrix[k]
            if isinstance(self.transition_state_matrix, list)
            else self.transition_state_matrix
        )

    def get_transition_input_matrix(self, k: int) -> np.ndarray:
        """k 시점의 입력 전이 행렬 B_k 반환."""
        return (
            self.transition_input_matrix[k]
            if isinstance(self.transition_input_matrix, list)
            else self.transition_input_matrix
        )

    def get_ineq_state_matrix(
        self,
        k: int,
    ) -> Union[None, np.ndarray, List[np.ndarray]]:
        """k 시점의 상태 제약 행렬 C_k 반환."""
        return (
            self.ineq_state_matrix[k]
            if isinstance(self.ineq_state_matrix, list)
            else self.ineq_state_matrix
        )

    def get_ineq_input_matrix(
        self,
        k: int,
    ) -> Union[None, np.ndarray, List[np.ndarray]]:
        """k 시점의 입력 제약 행렬 D_k 반환."""
        return (
            self.ineq_input_matrix[k]
            if isinstance(self.ineq_input_matrix, list)
            else self.ineq_input_matrix
        )

    def get_ineq_vector(self, k: int) -> np.ndarray:
        """k 시점의 제약 우변 벡터 e_k 반환."""
        return (
            self.ineq_vector[k]
            if isinstance(self.ineq_vector, list)
            else self.ineq_vector
        )

    # ------------------------------------------------------------------ #
    # 상태/목표/레퍼런스 trajectory 업데이트
    # ------------------------------------------------------------------ #
    def update_goal_state(self, goal_state: np.ndarray) -> None:
        
        self.goal_state = goal_state.flatten()

    def update_initial_state(self, initial_state: np.ndarray) -> None:

        self.initial_state = initial_state.flatten()

    def update_target_states(self, target_states: np.ndarray) -> None:

        self.target_states = target_states.flatten()

    # ------------------------------------------------------------------ #
    # For Debug
    # ------------------------------------------------------------------ #
    # def __repr__(self) -> str:
    #     return (
    #         "MPCProblem("
    #         f"goal_state={self.goal_state}, "
    #         f"ineq_input_matrix={self.ineq_input_matrix}, "
    #         f"ineq_state_matrix={self.ineq_state_matrix}, "
    #         f"ineq_vector={self.ineq_vector}, "
    #         f"initial_state={self.initial_state}, "
    #         f"input_dim={self.input_dim}, "
    #         f"nb_timesteps={self.nb_timesteps}, "
    #         f"stage_input_cost_weight={self.stage_input_cost_weight}, "
    #         f"stage_state_cost_weight={self.stage_state_cost_weight}, "
    #         f"state_dim={self.state_dim}, "
    #         f"terminal_cost_weight={self.terminal_cost_weight}, "
    #         f"transition_input_matrix={self.transition_input_matrix}, "
    #         f"transition_state_matrix={self.transition_state_matrix}"
    #         ")"
    #     )

    # ------------------------------------------------------------------ #
    # dynamics roll-out
    # ------------------------------------------------------------------ #
    def integrate(
        self,
        initial_state: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:

        X = np.zeros((self.nb_timesteps + 1, self.state_dim))
        X[0] = initial_state
        U = inputs

        for k in range(self.nb_timesteps):
            A_k = self.get_transition_state_matrix(k)
            B_k = self.get_transition_input_matrix(k)
            X[k + 1] = A_k.dot(X[k]) + B_k.dot(U[k])

        return X
