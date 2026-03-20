import numpy as np
import matplotlib.pyplot as plt


def maccormack_solver(nx, dt, nt, wave_type='square'):
    """
    MacCormack方法求解一维对流方程
    返回数值解、解析解和2-范数误差
    wave_type: 'square' 表示方波, 'gaussian' 表示高斯波
    """
    a = 1.0
    dx = 2.0 / (nx - 1)
    sigma = a * dt / dx
    x = np.linspace(0, 2, nx)

    # 初始条件
    if wave_type == 'square':
        # 方波初始条件
        u_corrected = np.zeros(nx)
        # 定义方波区域 [0.5, 1.0]
        mask = (x >= 0.5) & (x <= 1.0)
        u_corrected[mask] = 1.0
    else:  # gaussian
        u_corrected = np.exp(-100 * (x - 1) ** 2)

    # 时间演化
    for n in range(nt):
        u_pred_corr = np.copy(u_corrected)
        # 预测步
        u_pred_corr[:-1] = u_corrected[:-1] - sigma * (u_corrected[1:] - u_corrected[:-1])
        # 为右边界点添加预测步计算
        u_pred_corr[-1] = u_corrected[-1] - sigma * (u_corrected[-1] - u_corrected[-2])
        # 校正步
        u_corrected[1:] = 0.5 * (u_corrected[1:] + u_pred_corr[1:] - sigma * (u_pred_corr[1:] - u_pred_corr[:-1]))

    # 计算解析解
    t_final = nt * dt
    if wave_type == 'square':
        # 方波的解析解是初始波形向右平移 a*t_final
        u_exact = np.zeros(nx)
        # 方波区域 [0.5 + a*t_final, 1.0 + a*t_final]
        mask = (x >= 0.5 + a * t_final) & (x <= 1.0 + a * t_final)
        u_exact[mask] = 1.0
    else:  # gaussian
        u_exact = np.exp(-100 * (x - 1 - a * t_final) ** 2)

    # 计算2-范数误差
    error = np.sqrt(np.sum((u_corrected - u_exact) ** 2) * dx)

    return x, u_corrected, u_exact, error, dx


# 方波问题的数值解与解析解
nx = 200
dt = 0.005
nt = 200
x, u_num, u_exact, error, dx = maccormack_solver(nx, dt, nt, wave_type='square')

# 数值解与解析解
plt.figure(figsize=(12, 7))
plt.plot(x, u_num, 'b-', linewidth=2, label='numerical solution')
plt.plot(x, u_exact, 'r--', linewidth=2, label='exact solution')

plt.xlabel('x')
plt.ylabel('u')
plt.title('Square Wave: Comparison of Numerical and Exact Solutions')
plt.legend()
plt.grid(True)

plt.savefig('square_wave_solution.png')
plt.show()

print(f"方波问题的2-范数误差 (nx={nx}): {error:.6f}")

# 对比高斯波和方波的数值振荡
# 使用相同的参数
nx = 200
dt = 0.005
nt = 200

# 计算方波解
x, u_num_square, u_exact_square, error_square, dx = maccormack_solver(nx, dt, nt, wave_type='square')

# 计算高斯波解
x, u_num_gaussian, u_exact_gaussian, error_gaussian, dx = maccormack_solver(nx, dt, nt, wave_type='gaussian')

# 创建对比图
plt.figure(figsize=(14, 10))

# 方波
plt.subplot(2, 1, 1)
plt.plot(x, u_num_square, 'b-', linewidth=2, label='numerical solution')
plt.plot(x, u_exact_square, 'r--', linewidth=2, label='exact solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Square Wave: Numerical Oscillations')
plt.legend()
plt.grid(True)
plt.ylim(-0.5, 1.5)  # 调整y轴范围以更好地观察振荡

# 高斯波
plt.subplot(2, 1, 2)
plt.plot(x, u_num_gaussian, 'b-', linewidth=2, label='numerical solution')
plt.plot(x, u_exact_gaussian, 'r--', linewidth=2, label='exact solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Gaussian Wave: Smooth Solution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('wave_comparison.png')
plt.show()

# 分析不同网格分辨率下方波的数值振荡
nx_values = [50, 100, 200, 400]
dt = 0.005
nt = 200

plt.figure(figsize=(14, 10))

for i, nx in enumerate(nx_values):
    x, u_num, u_exact, error, dx = maccormack_solver(nx, dt, nt, wave_type='square')

    plt.subplot(2, 2, i + 1)
    plt.plot(x, u_num, 'b-', linewidth=2, label='numerical solution')
    plt.plot(x, u_exact, 'r--', linewidth=2, label='exact solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Square Wave: nx={nx}, Error={error:.4f}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.5, 1.5)  # 调整y轴范围以更好地观察振荡

plt.tight_layout()
plt.savefig('square_wave_resolution.png')
plt.show()

# 分析CFL数对数值振荡的影响
nx = 200
cfl_values = [0.2, 0.5, 0.8, 1.0]
nt = 200

plt.figure(figsize=(14, 10))

for i, cfl in enumerate(cfl_values):
    dx = 2.0 / (nx - 1)
    dt = cfl * dx
    x, u_num, u_exact, error, dx = maccormack_solver(nx, dt, nt, wave_type='square')

    plt.subplot(2, 2, i + 1)
    plt.plot(x, u_num, 'b-', linewidth=2, label='numerical solution')
    plt.plot(x, u_exact, 'r--', linewidth=2, label='exact solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Square Wave: CFL={cfl:.1f}, Error={error:.4f}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.5, 1.5)  # 调整y轴范围以更好地观察振荡

plt.tight_layout()
plt.savefig('square_wave_cfl.png')
plt.show()