import numpy as np
import time
import warnings
from boxQP import boxQP


def iLQR(DYNCST, x0, u0, u_lims):
    """
    Iterative LQR / iLQG solver.

    Parameters
    ----------
    DYNCST : callable
        DYNCST(x, u) -> x_next, c, fx, fu, cx, cu, cxx, cuu, cux
        - x : (n, T) or (n, K)   (state)
        - u : (m, T) or (m, K)   (control)
    x0 : np.ndarray
        Initial state trajectory or single initial state.
        Shape:
            - (n, 1)            : single state (no pre-rolled traj)
            - (n, N+1)          : pre-rolled trajectory states
    u0 : np.ndarray
        Initial control sequence, shape (m, N).
    u_lims : np.ndarray
        Control limits, shape (m, 2) for [lower, upper].

    Returns
    -------
    x : np.ndarray
        Optimized state trajectory, shape (n, N+1)
    u : np.ndarray
        Optimized control sequence, shape (m, N)
    cost : np.ndarray
        Stage costs over the trajectory, shape (N+1,)
    """

    # -------------------------------------------------------------------------
    # 1. 하이퍼파라미터
    # -------------------------------------------------------------------------
    opt = {
        "lims":           None,                      # control limits
        "parallel":       True,                      # parallel line-search
        "Alpha":          5 * 10 ** np.linspace(1, -3, 21),  # backtracking alphas
        "tolFun":         1e-7,                      # not 사용 중(주석 처리)
        "tolGrad":        1e-4,                      # gradient norm 종료 조건
        "maxIter":        1000,                      # 최대 반복 횟수
        "lambda":         1.0,                       # 초기 레귤러라이제이션 lambda
        "dlambda":        1.0,                       # lambda 스케일링 시작값
        "lambdaFactor":   1.6,                       # lambda 스케일 계수
        "lambdaMax":      1e10,                      # lambda 상한
        "lambdaMin":      1e-6,                      # lambda 하한
        "regType":        1,                         # 1: Quu+lambdaI, 2: Vxx+lambdaI
        "zMin":           0.0,                       # 최소 허용 reduction ratio
        "print":          2,                         # 0: none, 1: final, 2: iter
        "cost":           None,                      # pre-rolled cost (optional)
    }

    # -------------------------------------------------------------------------
    # 2. 기본 차원 설정
    # -------------------------------------------------------------------------
    n = x0.shape[0]      # state dimension
    m = u0.shape[0]      # control dimension
    N = u0.shape[1]      # horizon length (number of control steps)

    u = u0.copy()        # (m, N)
    opt["lims"] = u_lims

    verbosity = opt["print"]
    lamb = opt["lambda"]
    dlamb = opt["dlambda"]

    # -------------------------------------------------------------------------
    # 3. trace 초기화 (로깅용)
    # -------------------------------------------------------------------------
    trace_template = {
        "iter":         np.nan,
        "lambda":       np.nan,
        "dlambda":      np.nan,
        "cost":         np.nan,
        "alpha":        np.nan,
        "grad_norm":    np.nan,
        "improvement":  np.nan,
        "reduc_ratio":  np.nan,
        "time_derivs":  np.nan,
        "time_forward": np.nan,
        "time_backward": np.nan,
    }

    trace = np.tile(trace_template, np.minimum(opt["maxIter"], int(1e6)))
    trace[0]["iter"] = 1
    trace[0]["lambda"] = lamb
    trace[0]["dlambda"] = dlamb

    # -------------------------------------------------------------------------
    # 4. 초기 forward pass
    # -------------------------------------------------------------------------
    if x0.shape[1] == 1:
        # x0: single state (n, 1) -> 여러 alpha로 line-search 하면서 안정된 초기 traj 찾기
        diverge = True
        for alpha in opt["Alpha"]:
            # Alpha=[1]로 forward_pass 호출 (K=1)
            x_candidate, u_candidate, c_candidate = forward_pass(
                x0,
                alpha * u,
                L=None,
                x=None,
                du=None,
                Alpha=np.array([1.0]),
                DYNCST=DYNCST,
                lims=opt["lims"],
            )

            # x_candidate shape: (K, n, N+1) = (1, n, N+1)
            x_flat = x_candidate.reshape(n, N + 1)
            if np.all(np.abs(x_flat) < 1e8):
                x = x_flat
                u = u_candidate.reshape(m, N)
                cost = c_candidate.reshape(N + 1)
                diverge = False
                break

    elif x0.shape[1] == N + 1:
        # Pre-rolled trajectory가 주어진 경우
        x = x0
        diverge = False
        if opt["cost"] is None:
            raise ValueError("pre-rolled initial trajectory requires cost")
        else:
            cost = opt["cost"]
    else:
        raise ValueError("x0 must be (n,1) or (n,N+1)")

    trace[0]["cost"] = np.sum(cost)

    if diverge:
        # 초기 제어 시퀀스가 발산할 때
        Vx = np.nan
        Vxx = np.nan
        L = np.zeros((N, m, n))
        cost = None
        trace = trace[0]
        if verbosity > 0:
            print("\nEXIT: Initial control sequence caused divergence\n")
        return x, u, cost

    # -------------------------------------------------------------------------
    # 5. 반복 루프 준비
    # -------------------------------------------------------------------------
    flgChange = 1  # trajectory 변경 flag
    dcost = 0.0
    z = 0.0
    expected = 0.0

    print_head_every = 6
    last_head = print_head_every

    t_start = time.time()
    diff_t = np.zeros(opt["maxIter"])
    back_t = np.zeros(opt["maxIter"])
    fwd_t = np.zeros(opt["maxIter"])

    if verbosity > 0:
        print("\n=========== begin iLQR ===========\n")

    # -------------------------------------------------------------------------
    # 6. 메인 iLQR 반복
    # -------------------------------------------------------------------------
    for i in range(opt["maxIter"]):
        iter_idx = i + 1
        trace[i]["iter"] = iter_idx

        # ===== STEP 1. dynamics / cost 미분 =====
        if flgChange:
            t_diff = time.time()

            # u_sup 은 마지막에 dummy control 0 추가 (terminal cost만)
            u_sup = np.concatenate((u, np.zeros((m, 1))), axis=1)  # (m, N+1)
            # DYNCST(x, u_sup): x(n, N+1), u(m, N+1)
            _, _, fx, fu, cx, cu, cxx, cuu, cux = DYNCST(x, u_sup)

            trace[i]["time_derivs"] = time.time() - t_diff
            flgChange = 0

        # ===== STEP 2. backward pass (Ricatti) =====
        backPassDone = False

        while not backPassDone:
            t_back = time.time()

            diverge_bp, Vx, Vxx, k, L, dV = back_pass(
                fx, fu, cx, cu, cxx, cuu, cux,
                lamb,
                opt["regType"],
                opt["lims"],
                u,
            )

            trace[i]["time_backward"] = time.time() - t_back

            if diverge_bp:
                # Cholesky failure 등 -> lambda 증가
                if verbosity > 2:
                    print("Cholesky failed at timestep %d.\n" % diverge_bp)

                dlamb = np.maximum(dlamb * opt["lambdaFactor"], opt["lambdaFactor"])
                lamb = np.maximum(lamb * dlamb, opt["lambdaMin"])

                if lamb > opt["lambdaMax"]:
                    break
                continue

            backPassDone = True

        # gradient norm 계산
        # k: (m, N-1) -> 여기서는 du ≈ k
        g_norm = np.mean(np.max(np.abs(k) / (np.abs(u) + 1.0), axis=0))
        trace[i]["grad_norm"] = g_norm

        # gradient 기반 종료 조건
        if g_norm < opt["tolGrad"] and lamb < 1e-5:
            dlamb = np.minimum(dlamb / opt["lambdaFactor"], 1.0 / opt["lambdaFactor"])
            lamb = lamb * dlamb * (lamb > opt["lambdaMin"])
            if verbosity > 0:
                print("\nSUCCESS: gradient norm < tolGrad\n")
            break

        # ===== STEP 3. forward pass (line-search) =====
        fwdPassDone = False

        if backPassDone:
            t_fwd = time.time()

            if opt["parallel"]:
                # 병렬 line-search: 여러 alpha에 대해 동시에 forward_pass
                x_candidates, u_candidates, c_candidates = forward_pass(
                    x0,
                    u,
                    L=L,
                    x=x[:, 0:N],
                    du=k,
                    Alpha=opt["Alpha"],
                    DYNCST=DYNCST,
                    lims=opt["lims"],
                )

                # shapes:
                # x_candidates : (K, n, N+1)
                # u_candidates : (K, m, N)
                # c_candidates : (K, N+1)
                current_total_cost = np.sum(cost)
                new_total_costs = np.sum(c_candidates, axis=1)  # (K,)
                Dcost = current_total_cost - new_total_costs     # (K,)

                dcost = np.max(Dcost)
                best_idx = np.argmax(Dcost)

                alpha = opt["Alpha"][best_idx]
                expected = -alpha * (dV[0] + alpha * dV[1])

                if expected > 0:
                    z = dcost / expected
                else:
                    z = np.sign(dcost)
                    warnings.warn("non-positive expected reduction: should not occur")

                # original 코드: cost 감소 여부와 상관없이 항상 best step 채택
                fwdPassDone = True
                xnew = x_candidates[best_idx, :, :]
                unew = u_candidates[best_idx, :, :]
                costnew = c_candidates[best_idx, :]

            else:
                # 직렬 backtracking line-search
                for alpha in opt["Alpha"]:
                    x_candidate, u_candidate, c_candidate = forward_pass(
                        x0,
                        u + k * alpha,
                        L=L,
                        x=x[:, 0:N],
                        du=None,
                        Alpha=np.array([1.0]),
                        DYNCST=DYNCST,
                        lims=opt["lims"],
                    )

                    x_flat = x_candidate.reshape(n, N + 1)
                    u_flat = u_candidate.reshape(m, N)
                    c_flat = c_candidate.reshape(N + 1)

                    dcost = np.sum(cost) - np.sum(c_flat)
                    expected = -alpha * (dV[0] + alpha * dV[1])

                    if expected > 0:
                        z = dcost / expected
                    else:
                        z = np.sign(dcost)
                        warnings.warn("non-positive expected reduction: should not occur")

                    if z > opt["zMin"]:
                        fwdPassDone = True
                        xnew = x_flat
                        unew = u_flat
                        costnew = c_flat
                        break

            if not fwdPassDone:
                alpha = np.nan  # forward pass 실패 표시

            trace[i]["time_forward"] = time.time() - t_fwd

        # ===== STEP 4. step accept / reject, 출력 =====
        if verbosity > 1 and last_head == print_head_every:
            last_head = 0
            print(
                "%-12s%-12s%-12s%-12s%-12s%-12s"
                % ("iteration", "cost", "reduction", "expected", "gradient", "log10(lambda)")
            )

        if fwdPassDone:
            # 상태 출력
            if verbosity > 1:
                print(
                    "%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f"
                    % (iter_idx, np.sum(cost), dcost, expected, g_norm, np.log10(lamb))
                )
                last_head += 1

            # lambda 감소 (성공적인 step)
            dlamb = np.minimum(dlamb / opt["lambdaFactor"], 1.0 / opt["lambdaFactor"])
            lamb = lamb * dlamb * (lamb > opt["lambdaMin"])

            # 새 trajectory 채택
            x = xnew
            u = unew
            cost = costnew
            flgChange = 1

            # (원래 tolFun 종료 조건은 주석 처리 되어 있으므로 그대로 둠)

        else:
            # forward pass 실패 -> lambda 증가
            dlamb = np.maximum(dlamb * opt["lambdaFactor"], opt["lambdaFactor"])
            lamb = np.maximum(lamb * dlamb, opt["lambdaMin"])

            if verbosity > 1:
                print(
                    "%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f"
                    % (iter_idx, "NO STEP", dcost, expected, g_norm, np.log10(lamb))
                )
                last_head += 1

            if lamb > opt["lambdaMax"]:
                if verbosity > 0:
                    print("\nEXIT: lambda > lambdaMax\n")
                break

        # trace 업데이트
        trace[i]["lambda"] = lamb
        trace[i]["dlambda"] = dlamb
        trace[i]["alpha"] = alpha
        trace[i]["improvement"] = dcost
        trace[i]["cost"] = np.sum(cost)
        trace[i]["reduc_ratio"] = z

        diff_t[i] = trace[i]["time_derivs"]
        back_t[i] = trace[i]["time_backward"]
        fwd_t[i] = trace[i]["time_forward"]

    # -------------------------------------------------------------------------
    # 7. 종료 / 리포트
    # -------------------------------------------------------------------------
    if iter_idx == opt["maxIter"] and verbosity > 0:
        print("\nEXIT: Maximum iterations reached.\n")

    if iter_idx is not None:
        diff_t_total = np.sum(diff_t[~np.isnan(diff_t)])
        back_t_total = np.sum(back_t[~np.isnan(back_t)])
        fwd_t_total = np.sum(fwd_t[~np.isnan(fwd_t)])
        total_t = time.time() - t_start

        if verbosity > 0:
            print(
                "\n"
                f"iterations:   {iter_idx:<3d}\n"
                f"final cost:   {np.sum(cost):<12.7g}\n"
                f"final grad:   {g_norm:<12.7g}\n"
                f"final lambda: {lamb:<12.7e}\n"
                f"time / iter:  {1e3 * total_t / iter_idx:<5.0f} ms\n"
                f"total time:   {total_t:<5.2f} seconds, of which\n"
                f"  derivs:     {diff_t_total * 100 / total_t:<4.1f}%\n"
                f"  back pass:  {back_t_total * 100 / total_t:<4.1f}%\n"
                f"  fwd pass:   {fwd_t_total * 100 / total_t:<4.1f}%\n"
                f"  other:      {(total_t - diff_t_total - back_t_total - fwd_t_total) * 100 / total_t:<4.1f}% (graphics etc.)\n"
                "=========== end iLQG ===========\n"
            )
    else:
        raise ValueError("Failure: no iterations completed, something is wrong.")

    return x, u, cost


def forward_pass(x0, u, L, x, du, Alpha, DYNCST, lims):
    """
    Parallel forward rollout for multiple line-search alphas.

    Parameters
    ----------
    x0 : (n, 1)
        Initial state.
    u : (m, N)
        Nominal control sequence.
    L : (N, m, n) or None
        Feedback gains (from backward pass).
    x : (n, N) or None
        Nominal state trajectory (without last state).
    du : (m, N) or None
        Feedforward terms (k).
    Alpha : (K,)
        Line-search step sizes.
    DYNCST : callable
        Dynamics and cost function.
    lims : (m, 2) or None
        Control bounds.

    Returns
    -------
    xnew : (K, n, N+1)
    unew : (K, m, N)
    cnew : (K, N+1)
    """
    n = x0.shape[0]
    m, N = u.shape
    K = Alpha.shape[0]

    xnew = np.zeros((N + 1, n, K))
    xnew[0, :, :] = np.tile(x0, (1, K))
    unew = np.zeros((N, m, K))
    cnew = np.zeros((N + 1, K))

    for t in range(N):
        # 기본: nominal control 복제
        unew[t, :, :] = np.tile(u[:, t][:, None], (1, K))

        # feedforward term (du * alpha)
        if du is not None:
            du_t = du[:, t][:, None]           # (m,1)
            unew[t, :, :] += du_t @ Alpha[None, :]  # (m,K)

        # feedback term: L * (x - x_nominal)
        if L is not None:
            dx = xnew[t, :, :] - np.tile(x[:, t][:, None], (1, K))  # (n,K)
            unew[t, :, :] += L[t, :, :] @ dx                        # (m,K)

        # control saturation
        if lims is not None:
            unew[t, :, :] = np.clip(
                unew[t, :, :],
                lims[:, 0][:, None],
                lims[:, 1][:, None],
            )

        # dynamics / cost
        xnew[t + 1, :, :], cnew[t, :], *_ = DYNCST(
            xnew[t, :, :],              # (n,K)
            unew[t, :, :],              # (m,K)
        )

    # terminal cost (u=0)
    _, cnew[N, :], *_ = DYNCST(
        xnew[N, :, :],
        np.zeros((m, K)),
    )

    # time dimension을 마지막 축으로 옮기기: (K, n, N+1), (K, m, N), (K, N+1)
    xnew = np.transpose(xnew, (2, 1, 0))
    unew = np.transpose(unew, (2, 1, 0))
    cnew = np.transpose(cnew, (1, 0))

    return xnew, unew, cnew


def back_pass(fx, fu, cx, cu, cxx, cuu, cux, lamb, regType, lims, u):
    """
    Riccati backward pass.

    Parameters
    ----------
    fx, fu : dynamics Jacobians
        fx : (N, n, n)
        fu : (N, n, m)
    cx, cu : cost gradients
        cx : (n, N)
        cu : (m, N)
    cxx, cuu, cux : cost Hessians / cross terms
        cxx : (N, n, n)
        cuu : (N, m, m)
        cux : (N, m, n)
    lamb : float
        Regularization value.
    regType : int
        1 : Quu + lambdaI
        2 : Vxx + lambdaI
    lims : (m, 2) or None
        Control bounds.
    u : (m, N)
        Nominal control.

    Returns
    -------
    diverge : int
        0 if success, otherwise time index where PD failure occurred.
    Vx : (n, N)
    Vxx : (N, n, n)
    k : (m, N-1)
        Feedforward terms.
    K : (N-1, m, n)
        Feedback gains.
    dV : (2,)
        Expected cost reduction [first, second order term].
    """
    n, N = cx.shape
    m = cu.shape[0]

    k = np.zeros((m, N - 1))
    K = np.zeros((N - 1, m, n))
    Vx = np.zeros((n, N))
    Vxx = np.zeros((N, n, n))
    dV = np.zeros(2)

    # terminal conditions
    Vx[:, N - 1] = cx[:, N - 1]
    Vxx[N - 1, :, :] = cxx[N - 1, :, :]

    diverge = 0

    for t in reversed(range(N - 1)):
        fx_t = fx[t, :, :]    # (n,n)
        fu_t = fu[t, :, :]    # (n,m)

        cx_t = cx[:, t]       # (n,)
        cu_t = cu[:, t]       # (m,)

        cxx_t = cxx[t, :, :]  # (n,n)
        cuu_t = cuu[t, :, :]  # (m,m)
        cux_t = cux[t, :, :]  # (m,n)

        Vx_next = Vx[:, t + 1]       # (n,)
        Vxx_next = Vxx[t + 1, :, :]  # (n,n)

        # Q-function 미분들
        Qu = cu_t + fu_t.T @ Vx_next              # (m,)
        Qx = cx_t + fx_t.T @ Vx_next              # (n,)

        Qxx = cxx_t + fx_t.T @ Vxx_next @ fx_t    # (n,n)
        Quu = cuu_t + fu_t.T @ Vxx_next @ fu_t    # (m,m)
        Qux = cux_t + fu_t.T @ Vxx_next @ fx_t    # (m,n)

        # regularization
        Vxx_reg = Vxx_next + lamb * np.eye(n) * (regType == 2)
        Qux_reg = cux_t + fu_t.T @ Vxx_reg @ fx_t

        QuuF = cuu_t + fu_t.T @ Vxx_reg @ fu_t
        if regType == 1:
            QuuF = QuuF + lamb * np.eye(m)

        # ========== unconstrained (no control limit) ==========
        if lims is None or lims[0, 0] > lims[0, 1]:
            try:
                R = np.linalg.cholesky(QuuF)
            except np.linalg.LinAlgError:
                diverge = t + 1
                return diverge, Vx, Vxx, k, K, dV

            # Solve for k,K:  [Qu | Qux_reg]
            rhs = np.concatenate((Qu[:, None], Qux_reg), axis=1)  # (m, 1+n)
            kK = np.linalg.solve(-R, np.linalg.solve(R.T, rhs))   # (m, 1+n)

            k_t = kK[:, 0]
            K_t = kK[:, 1 : 1 + n]

        else:
            # ========== control-limited QP ==========
            lower = lims[:, 0] - u[:, t]
            upper = lims[:, 1] - u[:, t]

            # warm start : k[:, min(t+1, N-2)]
            warm_idx = min(t + 1, N - 2)
            warm_start = k[:, warm_idx]

            k_t, result, R, free = boxQP(QuuF, Qu, lower, upper, warm_start)

            if result < 1:
                diverge = t + 1
                return diverge, Vx, Vxx, k, K, dV

            K_t = np.zeros((m, n))
            if np.any(free):
                Lfree = np.linalg.solve(-R, np.linalg.solve(R.T, Qux_reg[free, :]))
                K_t[free, :] = Lfree

        # cost-to-go 업데이트
        dV[0] += k_t @ Qu
        dV[1] += 0.5 * (k_t @ Quu @ k_t)

        Vx[:, t] = (
            Qx
            + K_t.T @ Quu @ k_t
            + K_t.T @ Qu
            + Qux.T @ k_t
        )

        Vxx[t, :, :] = (
            Qxx
            + K_t.T @ Quu @ K_t
            + K_t.T @ Qux
            + Qux.T @ K_t
        )
        # 대칭화
        Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

        # 저장
        k[:, t] = k_t
        K[t, :, :] = K_t

    return diverge, Vx, Vxx, k, K, dV
