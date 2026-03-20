import numpy as np

def thomas(a, b, c, d):
    n = len(b)
    cp = np.zeros(n - 1, dtype=float)
    dp = np.zeros(n, dtype=float)

    # forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom

    denom = b[n - 1] - a[n - 2] * cp[n - 2]
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / denom

    # back substitution
    x = np.zeros(n, dtype=float)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

def book_sci(x, digits=3):
    if abs(x) < 1e-99:
        return f".{'0'*digits}E+00"

    sign_prefix = "-" if x < 0 else ""
    ax = abs(x)

    # Standard scientific notation: ax = m * 10^e, m in [1, 10)
    s = f"{ax:.{digits}E}"         # e.g. "9.997E-02"
    mant_str, exp_str = s.split("E")
    m = float(mant_str)
    e = int(exp_str)

    # Convert to book style: ax = (m/10) * 10^(e+1), mantissa in [0.1,1)
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
    ReD=5000.0,   # Re_D = rho*U*D/mu
    N=20,         # N+1 grid points (book example: 21 points)
    E=1.0,        # E = Δt* / (ReD*(Δy*)^2), book example uses E=1
    nsteps=360,
    output_steps=(12, 36, 60, 120, 240, 360)
):
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
    n_int = N - 1
    a = np.full(n_int - 1, A, dtype=float)  # subdiag
    b = np.full(n_int, B, dtype=float)      # diag
    c = np.full(n_int - 1, A, dtype=float)  # superdiag

    results = {}  # step -> (t*, u profile)

    for n in range(0, nsteps + 1):
        if n in output_steps:
            results[n] = (n * dt, u.copy())

        if n == nsteps:
            break

        # RHS K_j for j=1..N-1:
        # K_j = (1-E) u_j^n + (E/2)(u_{j+1}^n + u_{j-1}^n)
        K = np.zeros(n_int, dtype=float)
        for j in range(1, N):
            K[j - 1] = (1.0 - E) * u[j] + 0.5 * E * (u[j + 1] + u[j - 1])

        # boundary contribution for last interior equation (j=N-1):
        # A*u_N^{n+1} moved to RHS, with u_N^{n+1}=1
        K[-1] -= A * u[N]

        # solve for interior u^{n+1}
        u_int = thomas(a, b, c, K)

        # update full field
        u[1:N] = u_int
        u[0] = 0.0
        u[N] = 1.0

    return y, dt, results


def couette_analytic(y, t_star, ReD, n_terms=800):
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
        nsteps=max(steps_to_save),
        output_steps=steps_to_save
    )

    print(f"Δy* = {1/N:.6f},  Δt* = {dt:.6f}  (E={E}, ReD={ReD})")

    # 为了和课本表格逐行核对：默认只看 num 一列即可
    # 如需解析解对比，把 show_ana=True
    show_ana = True

    for n in steps_to_save:
        t_star, u_num = results[n]
        u_ana = couette_analytic(y, t_star, ReD, n_terms=800) if show_ana else None

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