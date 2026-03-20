import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# =========================================================
# Step 1: 全部亚声速等熵精确解 (7.78)(7.79) + (7.6)(7.7-7.9)
# =========================================================

def area_mach_relation(Ma, gamma=1.4):
    """等熵面积-马赫数关系 A/A* , 对应式(7.6)"""
    term1 = 1.0 / Ma
    term2 = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * Ma ** 2)
    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    return term1 * (term2 ** exponent)

def mach_from_area_ratio(A_ratio, gamma=1.4):
    """由 A/A* 反解 Ma（本算例全亚声速，取亚声速支）"""
    Ma_guess = 0.2
    Ma = fsolve(lambda Ma: area_mach_relation(Ma, gamma) - A_ratio, Ma_guess)[0]
    return Ma

def isentropic_relations(Ma, gamma=1.4):
    """等熵关系式(7.7)(7.8)(7.9): 已知 Ma 求 rho/rho0, T/T0, p/p0"""
    term = 1.0 + 0.5 * (gamma - 1.0) * Ma ** 2
    T_ratio = 1.0 / term
    rho_ratio = T_ratio ** (1.0 / (gamma - 1.0))
    p_ratio = T_ratio ** (gamma / (gamma - 1.0))
    return rho_ratio, T_ratio, p_ratio

# Gas parameter
gamma = 1.4

# =========================================================
# 网格/几何 (与课本7.4一致)
# =========================================================
L = 3.0
N = 31
dx = L / (N - 1)
x = np.linspace(0, L, N)

# 喷管面积分布 (7.80a)(7.80b) —— 以无量纲 x' 记号已省略
xprime = x / L
A = np.where(
    xprime <= 0.5,
    1.0 + 2.2 * (xprime - 0.5)**2,
    1.0 + 0.2223 * (xprime - 0.5)**2
)
# A = np.zeros_like(x)
for i in range(N):
    if x[i] <= 1.5:
        A[i] = 1.0 + 2.2 * (x[i] - 1.5) ** 2
    else:
        A[i] = 1.0 + 0.2223 * (x[i] - 1.5) ** 2

A_exit = A[-1]

# 出口压力比 (课本给定 pe/p0 = 0.93)
p0_exact = 1.0
rho0_exact = 1.0
T0_exact = 1.0
p_exit_ratio = 0.93
p_exit = p_exit_ratio * p0_exact

print("1、计算全亚声速等熵精确解ing")

# 由(7.78)用 pe/p0 先解出口马赫数 Me
# pe/p0 = (1 + (gamma-1)/2 * Me^2)^(-gamma/(gamma-1))
def pe_p0_from_M(M):
    term = 1.0 + 0.5*(gamma-1.0)*M**2
    return term**(-gamma/(gamma-1.0))

Me_guess = 0.2
Me = fsolve(lambda M: pe_p0_from_M(M) - p_exit_ratio, Me_guess)[0]

# 由(7.79)求 A*/(使用Me对应的 Ae/A*)
Ae_Astar = area_mach_relation(Me, gamma)  # Ae/A*
A_star = A_exit / Ae_Astar

rho_exact = np.zeros(N)
V_exact = np.zeros(N)
T_exact = np.zeros(N)
p_exact = np.zeros(N)
M_exact = np.zeros(N)

for i in range(N):
    A_ratio_local = A[i] / A_star    # A/A*
    M = mach_from_area_ratio(A_ratio_local, gamma)
    M_exact[i] = M
    rho_ratio, T_ratio, p_ratio = isentropic_relations(M, gamma)
    rho_exact[i] = rho0_exact * rho_ratio
    T_exact[i]   = T0_exact * T_ratio
    p_exact[i]   = p0_exact * p_ratio
    V_exact[i]   = M * np.sqrt(gamma * T_exact[i])

print("精确解计算完成!\n")

# =========================================================
# Step 2: MacCormack 数值解 (7.4.1~7.4.2)
# 控制方程与7.3相同
# 改动: 初值 (7.90) + 出口压力边界 (7.83~7.89)
# =========================================================

# 初始条件 (7.90a)(7.90b)(7.90c)
rho = 1.0 - 0.023 * x
T   = 1.0 - 0.009333 * x
V   = 0.05 + 0.11 * x

rho_pred = np.zeros(N)
V_pred   = np.zeros(N)
T_pred   = np.zeros(N)

# Courant number (课本取0.5)
C = 0.5

# 为画图保存若干时间点的场量
snap_steps = [0, 500, 1000, 5000]
snaps_rho, snaps_T, snaps_V = {}, {}, {}

print("开始MacCormack迭代求解(完全亚声速)...")
print(f"网格点数: {N}, dx = {dx:.4f}\n")

time_step = 0
max_steps = 5000

while time_step < max_steps:

    if time_step in snap_steps:
        snaps_rho[time_step] = rho.copy()
        snaps_T[time_step]   = T.copy()
        snaps_V[time_step]   = V.copy()

    rho_old = rho.copy()
    V_old   = V.copy()
    T_old   = T.copy()

    # 时间步长 (同7.3)
    a = np.sqrt(T)
    dt_local = C * dx / (a + V)
    dt = np.min(dt_local[1:-1])

    # ---------------------- 预测步 ----------------------
    for i in range(1, N-1):
        d_rho_dt_pred = (-rho[i] * (V[i+1] - V[i]) / dx
                         -rho[i] * V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx
                         -V[i] * (rho[i+1] - rho[i]) / dx)

        d_V_dt_pred = (-V[i] * (V[i+1] - V[i]) / dx
                       -(1.0/gamma) * ((T[i+1] - T[i]) / dx
                                       + (T[i]/rho[i]) * (rho[i+1] - rho[i]) / dx))

        d_T_dt_pred = (-V[i] * (T[i+1] - T[i]) / dx
                       -(gamma-1.0) * T[i] * ((V[i+1] - V[i]) / dx
                                              + V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx))

        rho_pred[i] = rho[i] + d_rho_dt_pred * dt
        V_pred[i]   = V[i]   + d_V_dt_pred   * dt
        T_pred[i]   = T[i]   + d_T_dt_pred   * dt

    # ---- 预测步边界条件 ----
    # 入口(亚声速入口): 与7.3一致 (7.70)(7.71)
    rho_pred[0] = 1.0
    T_pred[0]   = 1.0
    V_pred[0]   = 2.0*V_pred[1] - V_pred[2]

    # 出口(亚声速出口): 给定压力边界 (7.83)(7.84)
    # 采用(7.85)(7.86)(7.89)的组合
    T_pred[-1]   = 2.0*T_pred[-2] - T_pred[-3]      # (7.85)
    rho_pred[-1] = p_exit / T_pred[-1]              # (7.86) 使 rho_N*T_N = p_N
    V_pred[-1]   = 2.0*V_pred[-2] - V_pred[-3]      # (7.89)

    # ---------------------- 修正步 ----------------------
    rho_new = rho.copy()
    V_new   = V.copy()
    T_new   = T.copy()

    for i in range(1, N-1):
        d_rho_dt_corr = (-rho_pred[i] * (V_pred[i] - V_pred[i-1]) / dx
                         -rho_pred[i] * V_pred[i] * (np.log(A[i]) - np.log(A[i-1])) / dx
                         -V_pred[i] * (rho_pred[i] - rho_pred[i-1]) / dx)

        d_V_dt_corr = (-V_pred[i] * (V_pred[i] - V_pred[i-1]) / dx
                       -(1.0/gamma) * ((T_pred[i] - T_pred[i-1]) / dx
                                       + (T_pred[i]/rho_pred[i]) * (rho_pred[i] - rho_pred[i-1]) / dx))

        d_T_dt_corr = (-V_pred[i] * (T_pred[i] - T_pred[i-1]) / dx
                       -(gamma-1.0) * T_pred[i] * ((V_pred[i] - V_pred[i-1]) / dx
                                                   + V_pred[i] *
                                                   (np.log(A[i]) - np.log(A[i-1])) / dx))

        # 预测步导数(同7.3的平均格式)
        d_rho_dt_pred = (-rho[i] * (V[i+1] - V[i]) / dx
                         -rho[i] * V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx
                         -V[i] * (rho[i+1] - rho[i]) / dx)

        d_V_dt_pred = (-V[i] * (V[i+1] - V[i]) / dx
                       -(1.0/gamma) * ((T[i+1] - T[i]) / dx
                                       + (T[i]/rho[i]) * (rho[i+1] - rho[i]) / dx))

        d_T_dt_pred = (-V[i] * (T[i+1] - T[i]) / dx
                       -(gamma-1.0) * T[i] * ((V[i+1] - V[i]) / dx
                                              + V[i] *
                                              (np.log(A[i+1]) - np.log(A[i])) / dx))

        rho_new[i] = rho[i] + 0.5*(d_rho_dt_pred + d_rho_dt_corr)*dt
        V_new[i]   = V[i]   + 0.5*(d_V_dt_pred   + d_V_dt_corr  )*dt
        T_new[i]   = T[i]   + 0.5*(d_T_dt_pred   + d_T_dt_corr  )*dt

    # ---- 修正步边界条件 ----
    rho_new[0] = 1.0
    T_new[0]   = 1.0
    V_new[0]   = 2.0*V_new[1] - V_new[2]

    T_new[-1]   = 2.0*T_new[-2] - T_new[-3]
    rho_new[-1] = p_exit / T_new[-1]
    V_new[-1]   = 2.0*V_new[-2] - V_new[-3]

    rho, V, T = rho_new, V_new, T_new
    time_step += 1

    max_change = max(np.max(np.abs(rho - rho_old)),
                     np.max(np.abs(V   - V_old)),
                     np.max(np.abs(T   - T_old)))

    if time_step % 500 == 0:
        print(f"时间步: {time_step:5d}, 最大变化: {max_change:.2e}, dt: {dt:.6f}")

print("\n完成5000步计算!\n")

# ==== 补存最后一步快照，避免 snap_steps 含5000时报 KeyError ====
if max_steps in snap_steps:
    snaps_rho[max_steps] = rho.copy()
    snaps_T[max_steps]   = T.copy()
    snaps_V[max_steps]   = V.copy()


# 保存终态
rho_fin = rho.copy()
V_fin   = V.copy()
T_fin   = T.copy()
p_fin   = rho_fin * T_fin
a_fin   = np.sqrt(T_fin)
M_fin   = V_fin / a_fin
m_dot_fin = rho_fin * V_fin * A

# =========================================================
# Step 3: 输出稳态表格(对应课本7.4稳态对比表格式)
# =========================================================
print("稳态喷管流动参数(数值解 vs 精确解)")
print(f"{'I':>4s} {'x':>6s} {'A':>8s} {'rho_num':>9s} {'rho_ex':>9s} "
      f"{'V_num':>9s} {'V_ex':>9s} {'T_num':>9s} {'T_ex':>9s} "
      f"{'p_num':>9s} {'p_ex':>9s} {'Ma_num':>8s} {'Ma_ex':>8s}")

for i in range(N):
    print(f"{i+1:4d} {x[i]:6.2f} {A[i]:8.4f} "
          f"{rho_fin[i]:9.4f} {rho_exact[i]:9.4f} "
          f"{V_fin[i]:9.4f} {V_exact[i]:9.4f} "
          f"{T_fin[i]:9.4f} {T_exact[i]:9.4f} "
          f"{p_fin[i]:9.4f} {p_exact[i]:9.4f} "
          f"{M_fin[i]:8.4f} {M_exact[i]:8.4f}")

# =========================================================
# Step 4: 画图7.16 质量流量沿程随时间变化
# =========================================================
plt.figure(figsize=(9,6))
for st in snap_steps:
    rho_s = snaps_rho[st]
    V_s   = snaps_V[st]
    mdot_s = rho_s * V_s * A
    plt.plot(x, mdot_s, label=f"{st} Δt")
# 精确解点(黑点)
mdot_exact = rho_exact * V_exact * A
plt.scatter(x, mdot_exact, s=25, c='k', marker='o', label="exact")

plt.xlabel("x/L")
plt.ylabel("dimensionless mass flow, ρVA")
plt.title("Fig.7.16  mass flow variation with time (pe/p0=0.93)")
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig("Figure_7_16_massflow.png", dpi=300)
plt.show()

# =========================================================
# Step 5: 画图7.17 压力沿程随时间变化
# =========================================================
plt.figure(figsize=(9,6))
for st in snap_steps:
    rho_s = snaps_rho[st]
    T_s   = snaps_T[st]
    p_s = rho_s * T_s
    plt.plot(x, p_s, label=f"{st} Δt")
# 精确解点(黑点)
plt.scatter(x, p_exact, s=25, c='k', marker='o', label="exact")

plt.xlabel("x/L")
plt.ylabel("p/p0")
plt.title("Fig.7.17  pressure variation with time (pe/p0=0.93)")
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig("Figure_7_17_pressure.png", dpi=300)
plt.show()
