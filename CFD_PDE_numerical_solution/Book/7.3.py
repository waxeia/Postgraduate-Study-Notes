import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#步骤1: 计算等熵流动的精确解
def area_mach_relation(Ma, gamma=1.4):
    """等熵流动的面积-马赫数关系--公式7.6"""
    term1 = 1.0 / Ma
    term2 = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * Ma ** 2)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return term1 * (term2 ** exponent)#返回的就是A/A*

def mach_from_area_ratio(A_ratio, gamma=1.4, subsonic=True):#默认参数true，如果穿了false那就是超声速
    """从面积比 A/A* 反解马赫数"""
    if subsonic:
        Ma_guess = 0.5
        #数值求解器fsolve找到隐式方程的解
        Ma = fsolve(lambda Ma: area_mach_relation(Ma, gamma) - A_ratio, Ma_guess)[0]
    else:
        Ma_guess = 2.0
        Ma = fsolve(lambda Ma: area_mach_relation(Ma, gamma) - A_ratio, Ma_guess)[0]
    return Ma

def isentropic_relations(Ma, gamma=1.4):#isentropic等熵
    """等熵流动关系式--公式7.7、7.8、7.9--知Ma求ρ/ρ0、T/T0、p/p0"""
    term = 1.0 + 0.5 * (gamma - 1.0) * Ma ** 2
    T_ratio = 1.0 / term
    rho_ratio = T_ratio ** (1.0 / (gamma - 1.0))
    p_ratio = T_ratio ** (gamma / (gamma - 1.0))
    return rho_ratio, T_ratio, p_ratio

#气体参数
gamma = 1.4
C = 0.5
# 喷管参数
L = 3.0
N = 31
dx = L / (N - 1)
x = np.linspace(0, L, N)
# 喷管截面积
A = 1.0 + 2.2 * (x - 1.5) ** 2
A_throat = 1.0
print("1、计算亚声速-超声速等熵流动精确解ing")
rho0_exact = 1.0
T0_exact = 1.0
p0_exact = rho0_exact * T0_exact

rho_exact = np.zeros(N)
V_exact = np.zeros(N)
T_exact = np.zeros(N)
p_exact = np.zeros(N)
M_exact = np.zeros(N)

throat_idx = np.argmin(A)#即15，索引是从0开始的

for i in range(N):
    A_ratio = A[i] / A_throat
    if i <= throat_idx:
        M = mach_from_area_ratio(A_ratio, gamma, subsonic=True)
    else:
        M = mach_from_area_ratio(A_ratio, gamma, subsonic=False)

    M_exact[i] = M
    rho_ratio, T_ratio, p_ratio = isentropic_relations(M, gamma)

    rho_exact[i] = rho0_exact * rho_ratio
    T_exact[i] = T0_exact * T_ratio
    p_exact[i] = p0_exact * p_ratio
    V_exact[i] = M * np.sqrt(gamma * T_exact[i])
print("精确解计算完成!\n")

#步骤2: MacCormack数值求解
rho = 1.0 - 0.3146 * x
T = 1.0 - 0.2314 * x
V = (0.1 + 1.09 * x) * np.sqrt(T)

rho_pred = np.zeros(N)
V_pred = np.zeros(N)
T_pred = np.zeros(N)
print("开始MacCormack迭代求解...")
print(f"网格点数: {N}, dx = {dx:.4f}\n")
time_step = 0

while time_step < 1400:
    rho_old = rho.copy()
    V_old = V.copy()
    T_old = T.copy()

    # 计算时间步长
    a = np.sqrt(T)
    dt_local = C * dx / (a + V)
    dt = np.min(dt_local[1:-1])

    # 预测步
    for i in range(1, N - 1):
        d_rho_dt_pred = (-rho[i] * (V[i + 1] - V[i]) / dx
                         - rho[i] * V[i] * (np.log(A[i + 1]) - np.log(A[i])) / dx
                         - V[i] * (rho[i + 1] - rho[i]) / dx)
        d_V_dt_pred = (-V[i] * (V[i + 1] - V[i]) / dx
                       - (1.0 / gamma) * ((T[i + 1] - T[i]) / dx
                                          + (T[i] / rho[i]) * (rho[i + 1] - rho[i]) / dx))
        d_T_dt_pred = (-V[i] * (T[i + 1] - T[i]) / dx
                       - (gamma - 1.0) * T[i] * ((V[i + 1] - V[i]) / dx
                                                 + V[i] * (np.log(A[i + 1]) - np.log(A[i])) / dx))

        rho_pred[i] = rho[i] + d_rho_dt_pred * dt
        V_pred[i] = V[i] + d_V_dt_pred * dt
        T_pred[i] = T[i] + d_T_dt_pred * dt

    # 预测步边界条件
    rho_pred[0] = 1.0
    T_pred[0] = 1.0
    V_pred[0] = 2.0 * V_pred[1] - V_pred[2]
    rho_pred[N - 1] = 2.0 * rho_pred[N - 2] - rho_pred[N - 3]
    V_pred[N - 1] = 2.0 * V_pred[N - 2] - V_pred[N - 3]
    T_pred[N - 1] = 2.0 * T_pred[N - 2] - T_pred[N - 3]

    #修正步
    rho_new = rho.copy()
    V_new = V.copy()
    T_new = T.copy()

    for i in range(1, N - 1):
        d_rho_dt_corr = (-rho_pred[i] * (V_pred[i] - V_pred[i - 1]) / dx
                         - rho_pred[i] * V_pred[i] * (np.log(A[i]) - np.log(A[i - 1])) / dx
                         - V_pred[i] * (rho_pred[i] - rho_pred[i - 1]) / dx)
        d_V_dt_corr = (-V_pred[i] * (V_pred[i] - V_pred[i - 1]) / dx
                       - (1.0 / gamma) * ((T_pred[i] - T_pred[i - 1]) / dx
                                          + (T_pred[i] / rho_pred[i]) * (rho_pred[i] - rho_pred[i - 1]) / dx))
        d_T_dt_corr = (-V_pred[i] * (T_pred[i] - T_pred[i - 1]) / dx
                       - (gamma - 1.0) * T_pred[i] * ((V_pred[i] - V_pred[i - 1]) / dx
                                                      + V_pred[i] * (np.log(A[i]) - np.log(A[i - 1])) / dx))
        #重复
        d_rho_dt_pred = (-rho[i] * (V[i + 1] - V[i]) / dx
                         - rho[i] * V[i] * (np.log(A[i + 1]) - np.log(A[i])) / dx
                         - V[i] * (rho[i + 1] - rho[i]) / dx)
        d_V_dt_pred = (-V[i] * (V[i + 1] - V[i]) / dx
                       - (1.0 / gamma) * ((T[i + 1] - T[i]) / dx
                                          + (T[i] / rho[i]) * (rho[i + 1] - rho[i]) / dx))
        d_T_dt_pred = (-V[i] * (T[i + 1] - T[i]) / dx
                       - (gamma - 1.0) * T[i] * ((V[i + 1] - V[i]) / dx
                                                 + V[i] * (np.log(A[i + 1]) - np.log(A[i])) / dx))

        rho_new[i] = rho[i] + 0.5 * (d_rho_dt_pred + d_rho_dt_corr) * dt
        V_new[i] = V[i] + 0.5 * (d_V_dt_pred + d_V_dt_corr) * dt
        T_new[i] = T[i] + 0.5 * (d_T_dt_pred + d_T_dt_corr) * dt

    # 修正步边界条件
    rho_new[0] = 1.0
    T_new[0] = 1.0
    V_new[0] = 2.0 * V_new[1] - V_new[2]
    rho_new[N - 1] = 2.0 * rho_new[N - 2] - rho_new[N - 3]
    V_new[N - 1] = 2.0 * V_new[N - 2] - V_new[N - 3]
    T_new[N - 1] = 2.0 * T_new[N - 2] - T_new[N - 3]

    rho = rho_new
    V = V_new
    T = T_new

    time_step += 1

    max_change = max(
        np.max(np.abs(rho - rho_old)),
        np.max(np.abs(V - V_old)),
        np.max(np.abs(T - T_old))
    )

    if time_step % 100 == 0:
        print(f"时间步: {time_step:4d}, 最大变化: {max_change:.2e}, dt: {dt:.6f}")

print(f"\n完成1400步计算!\n")

# 计算最终结果
rho_1400 = rho.copy()
V_1400 = V.copy()
T_1400 = T.copy()
p_1400 = rho_1400 * T_1400
a_1400 = np.sqrt(T_1400)
M_1400 = V_1400 / a_1400
m_dot_1400 = rho_1400 * V_1400 * A

#3、输出表7.3格式的表格
print("表7.3: 喷管流动参数")
print(f"{'I':>4s} {'x':>8s} {'A/A*':>10s} {'ρ/ρ₀':>10s} {'V/a₀':>10s} {'T/T₀':>10s} {'p/p₀':>10s} {'Ma':>10s} {'ṁ':>10s}")
for i in range(N):
    A_ratio = A[i] / A_throat
    rho_ratio = rho_1400[i] / rho0_exact
    V_ratio = V_1400[i] / np.sqrt(gamma * T0_exact)
    T_ratio = T_1400[i] / T0_exact
    p_ratio = p_1400[i] / p0_exact
    print(f"{i + 1:4d} {x[i]:8.2f} {A_ratio:10.4f} {rho_ratio:10.4f} {V_ratio:10.4f} "
          f"{T_ratio:10.4f} {p_ratio:10.4f} {M_1400[i]:10.4f} {m_dot_1400[i]:10.4f}")

#4、绘制图7.12：一张图包含马赫数和密度两条线（含精确解）
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左侧Y轴 - 密度（使用数学文本渲染）
color1 = 'tab:blue'
ax1.set_xlabel('Distance along nozzle, x', fontsize=12)
ax1.set_ylabel(r'Density, $\rho/\rho_0$', fontsize=12, color=color1)

# 绘制密度的数值解（实线）和精确解（圆圈）
line1 = ax1.plot(x, rho_1400 / rho0_exact, 'b-', linewidth=2, label='Density (numerical)')
scatter1 = ax1.scatter(x, rho_exact / rho0_exact, s=40, facecolors='none', edgecolors='b',
                       linewidths=1.5, label='Density (exact)', zorder=5)

ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim([0, 3.0])
ax1.set_ylim([0, 1.2])
ax1.grid(True, alpha=0.3, linestyle='--')

# 右侧Y轴 - 马赫数
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Mach number, M', fontsize=12, color=color2)

# 绘制马赫数的数值解（实线）和精确解（圆圈）
line2 = ax2.plot(x, M_1400, 'r-', linewidth=2, label='Mach number (numerical)')
scatter2 = ax2.scatter(x, M_exact, s=40, facecolors='none', edgecolors='r',
                       linewidths=1.5, label='Mach number (exact)', zorder=5)

ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim([0, 3.6])

# 添加马赫数=1的参考线
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# 合并图例
lines = [line1[0], scatter1, line2[0], scatter2]
labels = ['Density (numerical)', 'Density (exact)', 'Mach number (numerical)', 'Mach number (exact)']
ax1.legend(lines, labels, loc='upper left', fontsize=9)

plt.title('nozzle: variation of dimensionless rho & Ma with dimensionless distance',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('Figure_7_12_Mach_Density.png', dpi=300, bbox_inches='tight')
plt.show()