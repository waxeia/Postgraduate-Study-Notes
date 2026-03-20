import numpy as np
import matplotlib.pyplot as plt

def maccormack_solver(nx, dt, nt):
    a = 1.0
    dx = 2.0 / (nx - 1)
    sigma = a * dt / dx
    x = np.linspace(0, 2, nx)

    # 初始条件
    u_corrected = np.exp(-100 * (x - 1) ** 2)

    for n in range(nt):
        u_pred_corr = np.copy(u_corrected)
        # predictor step
        u_pred_corr[:-1] = u_corrected[:-1] - sigma * (u_corrected[1:] - u_corrected[:-1])
        # the right point prediction step
        u_pred_corr[-1] = u_corrected[-1] - sigma * (u_corrected[-1] - u_corrected[-2])
        # corrector step
        u_corrected[1:] = 0.5 * (u_corrected[1:] + u_pred_corr[1:] - sigma * (u_pred_corr[1:] - u_pred_corr[:-1]))

    #计算解析解
    t_final = nt * dt
    u_exact = np.exp(-100 * (x - 1 - a * t_final) ** 2)

    # 计算2-范数误差
    error = np.sqrt(np.sum((u_corrected - u_exact) ** 2) * dx)
    return x, u_corrected, u_exact, error, dx

#设置不同的网格尺度
nx_values = np.array([160, 200, 240, 280, 320, 340, 360, 400])#这种方式更强大，可以减少循环
# nx_values = [20, 40, 80, 160, 200, 240, 280, 320, 340, 360, 400]
errors = []
dx_values = []

#计算每个网格尺度下的误差
for nx in nx_values:
    #调整时间步长
    dx = 2.0 / (nx - 1)
    dt = 0.2 * dx
    nt = int(1.0 / dt)#时间步数必须是整数（每次迭代推进一个 dt）

    x, u_num, u_exact, error, dx = maccormack_solver(nx, dt, nt)
    errors.append(error)
    dx_values.append(dx)
    log_dx = np.log10(dx)
    log_error = np.log10(error)
    print(f"nx={nx}, dx={dx:.4f}, 2-范数误差: {error:.6f}")
    print(f"log_dx={log_dx}, log_error={log_error}")

# 对数坐标系
plt.figure(figsize=(10, 6))
plt.loglog(dx_values, errors, 'bo-', linewidth=2, markersize=8)
plt.xlabel('dx')
plt.ylabel('L2 Error')
plt.title('analysis in log')
plt.grid(True, which="both", ls="--")

# 拟合直线
log_dx = np.log10(dx_values)
log_error = np.log10(errors)
coeffs = np.polyfit(log_dx, log_error, 1)
convergence_order = coeffs[0]  # 误差随dx减小而减小
plt.loglog(dx_values, 10 ** (coeffs[1]) * np.array(dx_values) ** (coeffs[0]), 'r--',
           label=f'Fit: Order = {convergence_order:.2f}')
plt.legend()
plt.savefig('convergence_analysis.png')
plt.show()
print(f"收敛阶: {convergence_order:.2f}")