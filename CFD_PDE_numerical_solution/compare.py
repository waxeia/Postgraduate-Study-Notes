import numpy as np
import matplotlib.pyplot as plt

a = 1.0
nx = 200
dx = 2.0 / (nx - 1)
dt = 0.01
nt = 100
sigma = a * dt / dx
x = np.linspace(0, 2, nx)

u_corrected = np.exp(-100 * (x - 1)**2)
for n in range(nt):
    u_pred_corr = np.copy(u_corrected)
    # 预测步
    u_pred_corr[:-1] = u_corrected[:-1] - sigma * (u_corrected[1:] - u_corrected[:-1])
    # --- 为右边界点添加预测步计算 ---
    u_pred_corr[-1] = u_corrected[-1] - sigma * (u_corrected[-1] - u_corrected[-2])
    # 校正步
    u_corrected[1:] = 0.5 * (u_corrected[1:] + u_pred_corr[1:] - sigma * (u_pred_corr[1:] - u_pred_corr[:-1]))

# --- 3. 计算解析解 ---
t_final = nt * dt
u_exact = np.exp(-100 * (x - 1 - a * t_final)**2)

# --- 4. 绘图并放大右边界 ---
plt.figure(figsize=(12, 7))
plt.plot(x, u_corrected, 'b-', linewidth=2, label='numerical solution')
plt.plot(x, u_exact, 'r--', linewidth=2, label='exact solution')

plt.xlabel('x')
plt.ylabel('u')
plt.title('compare graphic')
plt.legend()
plt.grid(True)

# 关键：放大右边界区域
plt.xlim(1.5, 2.0)
plt.ylim(0, 1.1)
plt.show()