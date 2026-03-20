import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib

# 完全关闭所有警告
warnings.filterwarnings('ignore')

# 配置matplotlib以避免字体警告
matplotlib.rcParams['axes.unicode_minus'] = False  # 用普通连字符代替Unicode负号
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 设置日志级别以隐藏字体相关的消息
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


def gauss_legendre_quadrature(f_values, x_points, n_gauss=3):
    """
    使用高斯-勒让德积分计算积分
    n_gauss: 每个子区间使用的高斯点数量（2, 3, 4, 5等）
    """
    sqrt3 = np.sqrt(3)
    sqrt5 = np.sqrt(5)
    sqrt6 = np.sqrt(6)
    sqrt10 = np.sqrt(10)
    sqrt30 = np.sqrt(30)
    sqrt70 = np.sqrt(70)

    gauss_data = {
        2: {
            'points': np.array([-1 / sqrt3, 1 / sqrt3]),
            'weights': np.array([1.0, 1.0])
        },
        3: {
            'points': np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]),
            'weights': np.array([5 / 9, 8 / 9, 5 / 9])
        },
        4: {
            'points': np.array([
                -np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),
                -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
                np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
                np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)
            ]),
            'weights': np.array([
                (18 - sqrt30) / 36,
                (18 + sqrt30) / 36,
                (18 + sqrt30) / 36,
                (18 - sqrt30) / 36
            ])
        },
        5: {
            'points': np.array([
                -np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3,
                -np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
                0.0,
                np.sqrt(5 - 2 * np.sqrt(10 / 7)) / 3,
                np.sqrt(5 + 2 * np.sqrt(10 / 7)) / 3
            ]),
            'weights': np.array([
                (322 - 13 * sqrt70) / 900,
                (322 + 13 * sqrt70) / 900,
                128 / 225,
                (322 + 13 * sqrt70) / 900,
                (322 - 13 * sqrt70) / 900
            ])
        }
    }

    gauss_points = gauss_data[n_gauss]['points']
    gauss_weights = gauss_data[n_gauss]['weights']

    integral = 0.0

    for i in range(len(x_points) - 1):
        x_left = x_points[i]
        x_right = x_points[i + 1]
        dx_local = x_right - x_left

        x_gauss = 0.5 * (x_right + x_left) + 0.5 * dx_local * gauss_points
        f_gauss = np.interp(x_gauss, x_points, f_values)
        local_integral = 0.5 * dx_local * np.sum(gauss_weights * f_gauss)
        integral += local_integral

    return integral


def maccormack_solver(nx, dt, nt, integration_method='gauss', n_gauss=3):
    """
    integration_method: 'rect', 'gauss'
    n_gauss: 高斯积分点数（2, 3, 4, 5）
    """
    a = 1.0
    dx = 2.0 / (nx - 1)
    sigma = a * dt / dx
    x = np.linspace(0, 2, nx)

    u_corrected = np.exp(-100 * (x - 1) ** 2)

    for n in range(nt):
        u_pred_corr = u_corrected.copy()
        u_pred_corr[:-1] = u_corrected[:-1] - sigma * (u_corrected[1:] - u_corrected[:-1])
        u_pred_corr[-1] = u_corrected[-1] - sigma * (u_corrected[-1] - u_corrected[-2])
        u_corrected[1:] = 0.5 * (u_corrected[1:] + u_pred_corr[1:] -
                                 sigma * (u_pred_corr[1:] - u_pred_corr[:-1]))

    t_final = nt * dt
    u_exact = np.exp(-100 * (x - 1 - a * t_final) ** 2)
    error_squared = (u_corrected - u_exact) ** 2

    if integration_method == 'rect':
        error = np.sqrt(np.sum(error_squared) * dx)
    elif integration_method == 'gauss':
        error = np.sqrt(gauss_legendre_quadrature(error_squared, x, n_gauss=n_gauss))
    else:
        raise ValueError(f"Unknown integration method: {integration_method}")

    return x, u_corrected, u_exact, error, dx


def main():
    nx = 150
    dt = 0.01
    nt = 100

    print("=" * 80)
    print("Single Point Test (nx=150):")
    print("=" * 80)

    methods = [
        ('rect', None, 'Rectangle'),
        ('gauss', 2, 'Gauss-2pt'),
        ('gauss', 3, 'Gauss-3pt'),
        ('gauss', 4, 'Gauss-4pt'),
        ('gauss', 5, 'Gauss-5pt')
    ]

    results = {}
    for method, n_gauss, name in methods:
        try:
            if method == 'gauss':
                x, u_num, u_exact, error, dx = maccormack_solver(
                    nx, dt, nt, integration_method=method, n_gauss=n_gauss)
            else:
                x, u_num, u_exact, error, dx = maccormack_solver(
                    nx, dt, nt, integration_method=method)
            results[name] = error
            print(f"{name:<20}: {error:.8f}")
        except Exception as e:
            print(f"{name:<20}: Failed - {str(e)}")

    # 绘图1
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, u_num, 'b-', linewidth=2, label='Numerical Solution')
    ax.plot(x, u_exact, 'r--', linewidth=2, label='Exact Solution')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u', fontsize=12)
    ax.set_title('Comparison of Numerical and Exact Solutions', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 2.0)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('solution_comparison_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 网格收敛性分析
    nx_values = np.array([160, 200, 240, 280, 320, 340, 360, 400])

    all_errors = {name: [] for _, _, name in methods}
    dx_values = []

    for nx in nx_values:
        dx = 2.0 / (nx - 1)
        dt = 0.2 * dx
        nt = int(1.0 / dt)
        dx_values.append(dx)

        for method, n_gauss, name in methods:
            try:
                if method == 'gauss':
                    _, _, _, error, _ = maccormack_solver(
                        nx, dt, nt, integration_method=method, n_gauss=n_gauss)
                else:
                    _, _, _, error, _ = maccormack_solver(
                        nx, dt, nt, integration_method=method)
                all_errors[name].append(error)
            except Exception:
                all_errors[name].append(np.nan)

    for i, nx in enumerate(nx_values):
        print(f"{nx:<8} {dx_values[i]:<12.6f} ", end='')
        for name in all_errors.keys():
            error_val = all_errors[name][i]
            if np.isnan(error_val):
                print(f"{'N/A':<18}", end='')
            else:
                print(f"{error_val:<18.8f}", end='')
        print()

    # 绘图2
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['green', 'blue', 'red', 'purple', 'orange', 'brown', 'pink']
    markers = ['s', 'o', '^', 'D', 'v', '<', '>']

    for (_, _, name), color, marker in zip(methods, colors, markers):
        errors = np.array(all_errors[name])
        valid_mask = ~np.isnan(errors)
        if not valid_mask.any():
            continue

        valid_dx = np.array(dx_values)[valid_mask]
        valid_errors = errors[valid_mask]

        ax.loglog(valid_dx, valid_errors, marker=marker, color=color,
                  linewidth=2, markersize=8, label=name, alpha=0.7)

        if len(valid_errors) > 1:
            log_valid_dx = np.log10(valid_dx)
            log_valid_error = np.log10(valid_errors)
            coeffs = np.polyfit(log_valid_dx, log_valid_error, 1)
            convergence_order = coeffs[0]

            ax.loglog(valid_dx,
                      10 ** (coeffs[1]) * valid_dx ** coeffs[0],
                      '--', color=color, linewidth=1.5, alpha=0.4)

            print(f"{name:<20}: Convergence order = {convergence_order:.4f}")

    ax.set_xlabel('Grid Spacing (dx)', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Convergence Analysis: Comparison of Integration Methods', fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=10, loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig('convergence_analysis_all_methods.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()