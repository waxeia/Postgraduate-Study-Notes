import numpy as np

#chasing method--->可以用我之前学习的LU分解思想来写代码！
def thomas(lower, diag, upper, b):  #diagonal主对角线
    n = len(diag)
    #LU分解decompose
    u_diag=np.zeros(n,dtype=float)
    l_lower=np.zeros(n-1,dtype=float)
    u_upper=np.zeros(n-1,dtype=float)#修改后的diagonal
#追
    #initial
    u_diag[0]=diag[0]
    if n>1:u_upper[0]=upper[0]
    #recurrence
    for i in range(1,n):
        l_lower[i-1]=lower[i-1]/u_diag[i-1]
        u_diag[i]=diag[i]-l_lower[i-1]*u_upper[i-1]
        if i<n-1:u_upper[i]=upper[i]#除去最后一行
    #LU---done
    #forward substitution Ly=b
    y=np.zeros(n,dtype=float)
    y[0]=b[0]#initial
    for i in range(1,n):y[i]=b[i]-l_lower[i-1]*y[i-1]#recurrence

#赶
    #backward substitution Ux=y
    x=np.zeros(n,dtype=float)
    x[n-1]=y[n-1]/u_diag[n-1]
    for i in range(n-2,-1,-1):#start stop step
        x[i]=(y[i]-u_upper[i]*x[i+1])/u_diag[i]
    return x

#课本科学计数法格式化
def book_sci(x,digits=3):#E前有三位数
    if abs(x)<1e-99:return f".000E+00"#<10^-99
    sign_prefix="-" if x<0 else ""#beautiful
    abs_x=abs(x)
    #标准科学计数法abs_x=m*10^e,m in[1,10),e是尾数
    s=f"{abs_x:.{digits}E}"#abs_x=0.09997 → s="9.997E-02"
    mantissa_str,exp_str=s.split("E")#尾数字符串与指数字符串
    m=float(mantissa_str)
    e=int(exp_str)#exp_str="-02" → e=-2
    #标准格式m×10^e = (m/10)×10^(e+1)课本格式
    m=m/10.0#float
    e=e+1
    m_round=round(m,digits)#四舍五入到digits位数,0.1235--->0.124
    if m_round>=1.0:#溢出
        m_round=m_round/10.0
        e=e+1
    frac=f"{m_round:.{digits}f}".split(".")[1]#m_round=0.100 → 格式化字符串"0.100" → 分割后["0","100"] → frac="100"   !!!!!!!!!!!!!!!!!
    exp_sign="+" if e>=0 else "-"
    exp_val=abs(e)
    return f"{sign_prefix}.{frac}E{exp_sign}{exp_val:02d}"

#core functions
def couette_cn(ReD=5000.0,N=20,E=1.0,sum_time_steps=360,output_steps=(12,36,60,120,240,360)):
    #ReD=ρUD/mu雷诺数；N:网格点数；E:控制时间步长的参数；sum_time_steps:总时间步数；output_steps:输出的时间步节点
    dy=1.0/N
    dt=E*ReD*(dy**2)#控制E参数就是控制时间步长dt
    y=np.linspace(0.0,1.0,N+1)#无量纲y坐标
    u=np.zeros(N+1,dtype=float)#初始条件u(y,0)=0,u是无量纲速度
    u[0]=0.0#下板u(0,t)=0，即下边界条件
    u[N]=1.0#上板u(1,t)=1，即上边界条件
    #Crank-Nicolson系数矩阵的三对角矩阵
    A=-E/2.0
    B=1+E
    dimension=N-1#未知数个数--19
    lower=np.full(dimension-1,A,dtype=float)#下对角线
    diag=np.full(dimension,B,dtype=float)#主对角线
    upper=np.full(dimension-1,A,dtype=float)#上对角线
    results={}#创建一个空字典，键值对，key:时间步；value:(无量纲时间，对应速度剖面)
    for n in range(0,sum_time_steps+1):
        if n in output_steps:
            results[n]=(n*dt,u.copy())#n是key，等号右边是value
        if n==sum_time_steps:break
        K=np.zeros(dimension,dtype=float)#右端单列向量---每一个时间步有不同的K，对应不同的速度剖面
        for j in range(1,N):#对应内点2-->N
            K[j-1]=(1.0-E)*u[j]+0.5*E*(u[j+1]+u[j-1])
        #更新最后一个内点
        K[-1]-=A*u[N]
        #求解下一个时间步的速度剖面
        x=thomas(lower,diag,upper,K)
        u[1:N]=x#更新内点，边界保持不变
    return y,dt,results

#exact solution
def couette_exact(y,t_star,ReD,n_terms=800):
    #y无量纲纵坐标；t_star无量纲时间；n_terms级数项数
    u_exact=np.asarray(y,dtype=float)#将输入的(非)NumPy数组类型y转换为NumPy数组
    base=y.copy()#基础线性剖面
    s=np.zeros_like(y)#与y同类型but为0
    for m in range(1,n_terms+1):#极数项循环求和
        sign=-1.0 if (m%2==1) else 1.0#级数项中的符号因子(-1)^m
        s += (sign / m) * np.sin(m * np.pi * y) * np.exp(-(m ** 2) * (np.pi ** 2) * t_star / ReD)
    return base+(2.0/np.pi)*s

if __name__=="__main__":
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
            print(" j   y/D      u/u_e (numerical)   u/u_e (exact)")
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