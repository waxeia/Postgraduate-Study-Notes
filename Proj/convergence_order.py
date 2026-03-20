import numpy as np
import matplotlib.pyplot as plt

def maccormack_solver(nx, dt, nt):
    a=1.0#对流速度，一维线性对流方程中的a
    dx=2.0/(nx-1)
    sigma=a*dt/dx #CFL库朗数
    x=np.linspace(0,2,nx)#等差数列，返回x坐标值数组
    #初始高斯波
    u_y=np.exp(-100*(x-1)**2)#返回y坐标值数组
    for n in range(nt):#nt总时间区间离散化
        u_y_pred=np.copy(u_y)
        #预测步
        u_y_pred[:-1]=u_y[:-1]-sigma*(u_y[1:]-u_y[:-1])#不包含最后一个点-1处
        #补充：切片利用了 NumPy 的向量化特性，避免了显式写循环
        #最右侧点的预测步处理
        u_y_pred[-1]=u_y[-1]-sigma*(u_y[-1]-u_y[-2])
        #校正步
        u_y[1:]=0.5*(u_y[1:]+u_y_pred[1:]-sigma*(u_y_pred[1:]-u_y_pred[:-1]))#切片：左闭右开，并且左边下标从0开始的
        #TODO疑惑：u_y[0]作为边界点，其值由物理问题的边界条件决定。 u_y[0]没有被更新

    #解析解--简单的平移
    t_sum=nt*dt
    u_exact=np.exp(-100*(x-1-a*t_sum)**2)

    #2-范数误差-矩形积分
    err=np.sqrt(np.sum((u_y-u_exact)**2)*dx)

    return x,u_y,u_exact,err,dx#返回的项挺多

#pyplot是matplotlib库中的子库
plt.figure(figsize=(12, 7))#创建长12inch宽7inch的画布

#设置不同网格尺度
nx_vals=np.array([160,200,240,280,320,340,360,400])
errs=[]
dx_vals=[]
log_errs=[]
log_dx_vals=[]
#计算不同网格尺度的误差
for nx in nx_vals:
    dx=2.0/(nx-1)
    dt=0.2*dx#调整时间步长，保持CFL数不变
    nt=int(1.0/dt)#时间步数必须是整数（每次迭代推进一个 dt）
    #调用maccormack_solver
    x,u_y,u_exact,err,dx=maccormack_solver(nx,dt,nt)
    errs.append(err)
    dx_vals.append(dx)
    #取以10为底的对数值
    log_dx_vals.append(np.log10(dx))
    log_errs.append(np.log10(err))
    #每循环一次就打印一次，规律：网格分的越细，误差越小
    print(f"nx={nx}, dx={dx:.4f}, L2误差（矩形积分）为{err:.5f}")#识别为f-string这种字符串并格式化为保留4位小数的浮点数

#画图：对数坐标系下误差收敛图
plt.figure(figsize=(10,6))#长10inch 宽6inch
#b:蓝色，o:数据点用圆形标记，-:用直线连接数据点；linewidth=2：连接线的粗细；markersize=8：数据点标记的大小，便于清晰展示每个网格下的具体数据
plt.loglog(dx_vals,errs,'bo-',linewidth=2,markersize=8)#绘制双对数坐标系（x 轴和 y 轴均为对数尺度）
plt.xlabel("dx",x=1.0)#x=1.0:标签的右边缘与轴的右边界对齐
plt.ylabel("L2 error--rectangle integral",y=0.8)#y=0.8:标签的上边缘与轴的上边界对齐
plt.title("L2 Error in Log-Log")
plt.grid(True, which="both", ls="--")
#双对数坐标系下已经画好了--描点、连线
#拟合一条直线--最小二乘法
coeffs=np.polyfit(log_dx_vals,log_errs,1)#系数：斜率和截距
# 对数据 (x,y) 进行拟合，degree=1 表示拟合成一次函数
# 返回结果 coeffs 是一个长度为 2 的数组：[斜率, 截距]，对应直线方程 y = 斜率 × x + 截距。
# 提取斜率：收敛阶 p（log_error = p×log_dx + C）
convergence_order=coeffs[0]#数组的第一个元素
#画出这条拟合的直线
plt.loglog(dx_vals, 10 ** (coeffs[1]) * np.array(dx_vals) ** (coeffs[0]), 'r--',
           label=f'Fit: Order = {convergence_order:.2f}')


plt.savefig('try.png')
print(f"收敛阶: {convergence_order:.2f}")





