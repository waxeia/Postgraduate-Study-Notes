import numpy as np
import matplotlib.pyplot as plt
#fsolve：要导入的具体对象，optimize 子模块中用于求解非线性方程组的 fsolve 函数
from scipy.optimize import fsolve#精准导入
import funs_seven_point_three as f73

#1、等熵流动的精确解
#参数
gamma=1.4#比热比
C=0.5#Courant数:a*dt/dx
L=3.0#喷管长度
N=31#网格点数，try 61
dx=L/(N-1)#网格间距
x=np.linspace(0,L,N)#网格点横坐标构成的数组
A=1+2.2*(x-1.5)**2#喷管截面积--抛物线，x是一个数组，A也是一个数组
A_throat=1#喷管喉部截面积
print("1、计算亚声速-超声速等熵流动精确解ing")
rho0_exact=1#精确解下的初始密度且无量纲化
T0_exact=1#精确解下的初始温度且无量纲化
p0_exact=1#状态方程rho0_exact*T0_exact
#精确解下，初始的零数组
rho_exact=np.zeros(N)#每一个网格点处的rho
V_exact=np.zeros(N)
T_exact=np.zeros(N)
p_exact=np.zeros(N)
Ma_exact=np.zeros(N)
throat_idx=np.argmin(A)#截面数组A的最小值的索引，即喉部位置的索引
#截面带动流动（参数）
for i in range(N):
    A_ratio=A[i]/A_throat
    if i<=throat_idx:#TODO:喉部位置该用亚声速还是超声速？
        Ma = f73.mach_solved_from_area(A_ratio, gamma, subsonic=True)
    else:
        Ma = f73.mach_solved_from_area(A_ratio, gamma, subsonic=False)
    Ma_exact[i]=Ma#在 Python 中，if/else、for/while等控制流结构并不会创建新的局部作用域，与C++不同
    rho_ratio,T_ratio,p_ratio=f73.mach_represents_T_rho_p_ratio(Ma,gamma)
    rho_exact[i]=rho0_exact*rho_ratio
    T_exact[i]=T0_exact*T_ratio
    #无量纲静温度=无量纲总温度（参考值 1.0）×无量纲温度比T/T0
    p_exact[i]=p0_exact*p_ratio
    V_exact[i]=Ma*np.sqrt(gamma*T_exact[i])#TODO:无量纲声速
print("精确解完成！")

#2、Maccormack数值解
print("Maccormack迭代ing\n网格点数:{},dx={:.4f}".format(N,dx))
#迭代初始值，经验性猜测
rho=1-0.3146*x#x是数组
T=1-0.2314*x
V=(0.1+1.09*x)*np.sqrt(T)
rho_pred=np.zeros(N)
V_pred=np.zeros(N)
T_pred=np.zeros(N)
time_step=0
while time_step<1400:
    rho_copy=rho.copy()#copy的旧值
    V_copy=V.copy()
    T_copy=T.copy()
    #时间步长
    c=np.sqrt(T)#TODO:无量纲声速
    dt_local=C*dx/(c+V)#c+V都是标准的无量纲，即局部时间步长dt_local=dt
    #dt_local是一个数组！！！
    dt=np.min(dt_local[1:-1])#dt在随循环变化
    drho_dt_pred_array=np.zeros(N)
    dV_dt_pred_array=np.zeros(N)
    dT_dt_pred_array=np.zeros(N)
    #预测步
    for i in range(1,N-1):#[1,N-1)，入口点是固定的，出口点前向差分不了
        drho_dt_pred=(-rho[i]*(V[i+1]-V[i])/dx-rho[i]*V[i]*(np.log(A[i+1])-np.log(A[i]))/dx)-V[i]*(rho[i+1]-rho[i])/dx
        drho_dt_pred_array[i]=drho_dt_pred
        dV_dt_pred=-V[i]*(V[i+1]-V[i])/dx-1/gamma*((T[i+1]-T[i])/dx+T[i]/rho[i]*(rho[i+1]-rho[i])/dx)
        dV_dt_pred_array[i]=dV_dt_pred
        dT_dt_pred=-V[i]*(T[i+1]-T[i])/dx-(gamma-1)*T[i]*((V[i+1]-V[i])/dx+V[i]*(np.log(A[i+1])-np.log(A[i]))/dx)
        dT_dt_pred_array[i]=dT_dt_pred
        #迭代累加
        rho_pred[i]=rho[i]+drho_dt_pred*dt
        V_pred[i]=V[i]+dV_dt_pred*dt
        T_pred[i]=T[i]+dT_dt_pred*dt
    #预测步边界条件：入口和出口
    rho_pred[0]=1
    T_pred[0]=1
    V_pred[0]=2*V_pred[1]-V_pred[2]#线性外插，为什么不是固定值？V的无量钢化基准是a0
    rho_pred[N-1]=2*rho_pred[N-2]-rho_pred[N-3]
    V_pred[N-1]=2*V_pred[N-2]-V_pred[N-3]
    T_pred[N-1]=2*T_pred[N-2]-T_pred[N-3]
    #修正步
    rho_new=rho.copy()
    V_new=V.copy()
    T_new=T.copy()
    for i in range(1,N-1):
        drho_dt_corr=-rho_pred[i]*(V_pred[i]-V_pred[i-1])/dx-rho_pred[i]*V_pred[i]*(np.log(A[i])-np.log(A[i-1]))/dx-V_pred[i]*(rho_pred[i]-rho_pred[i-1])/dx
        dV_dt_corr=-V_pred[i]*(V_pred[i]-V_pred[i-1])/dx-1/gamma*((T_pred[i]-T_pred[i-1])/dx+T_pred[i]/rho_pred[i]*(rho_pred[i]-rho_pred[i-1])/dx)
        dT_dt_corr=-V_pred[i]*(T_pred[i]-T_pred[i-1])/dx-(gamma-1)*T_pred[i]*((V_pred[i]-V_pred[i-1])/dx+V_pred[i]*(np.log(A[i])-np.log(A[i-1]))/dx)
        #预测步和修正步对时间导数的平均值+迭代累加
        rho_new[i]=rho[i]+0.5*(drho_dt_pred_array[i]+drho_dt_corr)*dt
        V_new[i]=V[i]+0.5*(dV_dt_pred_array[i]+dV_dt_corr)*dt
        T_new[i]=T[i]+0.5*(dT_dt_pred_array[i]+dT_dt_corr)*dt
    #修正步边界条件
    rho_new[0]=1
    T_new[0]=1
    V_new[0]=2*V_new[1]-V_new[2]
    rho_new[N-1]=2*rho_new[N-2]-rho_new[N-3]
    V_new[N-1]=2*V_new[N-2]-V_new[N-3]
    T_new[N-1]=2*T_new[N-2]-T_new[N-3]
    #更新参数
    rho=rho_new
    V=V_new
    T=T_new
    #时间推进1
    time_step+=1
    #控制迭代稳定的精度
    max_change=max(np.max(np.abs(rho-rho_copy)),np.max(np.abs(V-V_copy)),np.max(np.abs(T-T_copy)))#np.max避免了循环
    #稳定监控
    if time_step%100==0:
        print("时间步:{:4d},最大变化:{:.2e},dt:{:.6f}".format(time_step,max_change,dt))#.2e:科学计数法显示数值，且尾数部分保留2位小数
print("time_step=1400步迭代完成！")
#计算结果--array
rho_1400=rho.copy()
V_1400=V.copy()
T_1400=T.copy()
#无量纲状态方程
p_1400=rho_1400*T_1400
a_1400=np.sqrt(T_1400)
Ma_1400=V_1400/a_1400
m_dot_1400=rho_1400*V_1400*A#质量流量

#3、打印7.3对照表格
print("表7.3：喷管流动参数")
print(f"{'I':>4s}{'x':>8s}{'A/A*':>10s}{'ρ/ρ₀':>10s}{'V/a0':>10s}{'T/T0':>10s}{'p/p0':>10s}{'Ma':>10s}{'m':>10s}")
for i in range(N):
    A_ratio=A[i]/A_throat
    rho_ratio=rho_1400[i]/rho0_exact
    V_ratio=V_1400[i]/np.sqrt(gamma*T0_exact)#无量纲声速取哪一个形式
    T_ratio=T_1400[i]/T0_exact
    p_ratio=p_1400[i]/p0_exact
    print(f"{i+1:4d}{x[i]:8.3f}{A_ratio:10.3f}{rho_ratio:10.3f}{V_ratio:10.3f}{T_ratio:10.3f}{p_ratio:10.3f}{Ma_1400[i]:10.3f}{m_dot_1400[i]:10.3f}")

#4、绘图
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
line2 = ax2.plot(x, Ma_1400, 'r-', linewidth=2, label='Mach number (numerical)')
scatter2 = ax2.scatter(x, Ma_exact, s=40, facecolors='none', edgecolors='r',
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