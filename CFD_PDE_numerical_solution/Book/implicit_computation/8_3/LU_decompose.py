import numpy as np

def thomas(lower, diag, upper, b):
    n = len(diag)

    # LU分解
    # U的主对角线
    u_diag = np.zeros(n, dtype=float)
    # L的下对角线 (主对角线默认为1)
    l_lower = np.zeros(n - 1, dtype=float)
    # U的上对角线就是修改后的c
    u_upper = np.zeros(n - 1, dtype=float)

    # 第一行：u_diag[0] = b[0]
    u_diag[0] = diag[0]
    if n > 1:
        u_upper[0] = upper[0]

    # 从第二行开始分解
    for i in range(1, n):
        # l_lower[i-1] = a[i-1] / u_diag[i-1]
        l_lower[i - 1] = lower[i - 1] / u_diag[i - 1]
        # u_diag[i] = b[i] - l_lower[i-1] * u_upper[i-1]
        u_diag[i] = diag[i] - l_lower[i - 1] * u_upper[i - 1]
        # 如果不是最后一行，更新上对角线
        if i < n - 1:
            u_upper[i] = upper[i]

    # 前向替换：求解 Ly = d
    # L是下双对角矩阵，主对角线为1
    y = np.zeros(n, dtype=float)
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - l_lower[i - 1] * y[i - 1]

    # 后向替换：求解 Ux = y
    # U是上双对角矩阵
    x = np.zeros(n, dtype=float)
    x[n - 1] = y[n - 1] / u_diag[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - u_upper[i] * x[i + 1]) / u_diag[i]

    return x

def book_sci(x, digits=3):
    if abs(x) < 1e-99:
        return f".000E+00"

    sign_prefix = "-" if x < 0 else ""
    abs_x = abs(x)

    # Standard scientific notation: abs_x = m * 10^e, m in [1, 10)
    s = f"{abs_x:.{digits}E}"  # e.g. "9.997E-02"
    mantissa_str, exp_str = s.split("E")
    m = float(mantissa_str)
    e = int(exp_str)

    # Convert to book style: abs_x = (m/10) * 10^(e+1), mantissa in [0.1,1)
    m = m / 10.0
    e = e + 1

    # Round mantissa to requested digits (THIS is where carry can happen)
    m_round = round(m, digits)

    # If rounding pushed it to 1.000, renormalize: 1.000 * 10^e -> 0.100 * 10^(e+1)
    if m_round >= 1.0:
        m_round /= 10.0
        e += 1

    # Build ".xxx"
    frac = f"{m_round:.{digits}f}".split(".")[1]  # keep digits after dot
    exp_sign = "+" if e >= 0 else "-"
    exp_val = abs(e)

    return f"{sign_prefix}.{frac}E{exp_sign}{exp_val:02d}"


def couette_cn(
        ReD=5000.0,  # Re_D = rho*U*D/mu
        N=20,  # N+1 grid points (book example: 21 points)
        E=1.0,  # E = Δt* / (ReD*(Δy*)^2), book example uses E=1
        sum_time_steps=360,
        output_steps=(12, 36, 60, 120, 240, 360)):
    dy = 1.0 / N
    dt = E * ReD * (dy ** 2)  # from E = dt/(ReD*dy^2)
    y = np.linspace(0.0, 1.0, N + 1)

    # initial condition
    u = np.zeros(N + 1, dtype=float)
    u[0] = 0.0
    u[N] = 1.0

    # CN coefficients
    A = -E / 2.0
    B = 1.0 + E

    # interior unknowns: j=1..N-1 => size N-1
    dimension = N - 1
    lower = np.full(dimension - 1, A, dtype=float)  # subdiag
    diag = np.full(dimension, B, dtype=float)  # diag
    upper = np.full(dimension - 1, A, dtype=float)  # superdiag

    results = {}  # step -> (t*, u profile)

    for n in range(0, sum_time_steps + 1):
        if n in output_steps:
            results[n] = (n * dt, u.copy())

        if n == sum_time_steps:
            break

        # RHS K_j for j=1..N-1:
        # K_j = (1-E) u_j^n + (E/2)(u_{j+1}^n + u_{j-1}^n)
        K = np.zeros(dimension, dtype=float)
        for j in range(1, N):
            K[j - 1] = (1.0 - E) * u[j] + 0.5 * E * (u[j + 1] + u[j - 1])
        # boundary contribution for last interior equation (j=N-1):
        # A*u_N^{n+1} moved to RHS, with u_N^{n+1}=1
        K[-1] -= A * u[N]
        # solve for interior u^{n+1}
        x = thomas(lower, diag, upper, K)
        # update full field
        u[1:N] = x
        # u[0] = 0.0
        # u[N] = 1.0
    return y, dt, results

def couette_exact(y, t_star, ReD, n_terms=800):
    y = np.asarray(y, dtype=float)
    base = y.copy()
    s = np.zeros_like(y)
    for m in range(1, n_terms + 1):
        sign = -1.0 if (m % 2 == 1) else 1.0  # (-1)^m
        s += (sign / m) * np.sin(m * np.pi * y) * np.exp(-(m ** 2) * (np.pi ** 2) * t_star / ReD)
    return base + (2.0 / np.pi) * s

if __name__ == "__main__":
    # match book example
    ReD = 5000.0
    N = 20
    E = 1.0
    steps_to_save = (12, 36, 60, 120, 240, 360)

    y, dt, results = couette_cn(
        ReD=ReD, N=N, E=E,
        sum_time_steps=max(steps_to_save),
        output_steps=steps_to_save
    )

    print(f"Δy* = {1 / N:.6f},  Δt* = {dt:.6f}  (E={E}, ReD={ReD})")

    # 为了和课本表格逐行核对：默认只看 num 一列即可
    # 如需解析解对比，把 show_ana=True
    show_ana = True

    for n in steps_to_save:
        t_star, u_num = results[n]
        u_ana = couette_exact(y, t_star, ReD, n_terms=800) if show_ana else None
        print("\n" + "=" * 72)
        print(f"n = {n:4d} steps,  t* = n·Δt* = {t_star:.6f}")
        if show_ana:
            print(" j   y/D      u/u_e (num)   u/u_e (ana)")
        else:
            print(" j   y/D      u/u_e")

        for j in range(len(y)):
            y_str = f"{y[j]:.2f}"
            num_s = book_sci(u_num[j], digits=3)

            if show_ana:
                ana_s = book_sci(u_ana[j], digits=3)
                print(f"{j:2d}  {y_str:>4s}     {num_s:>9s}     {ana_s:>9s}")
            else:
                print(f"{j:2d}  {y_str:>4s}     {num_s:>9s}")