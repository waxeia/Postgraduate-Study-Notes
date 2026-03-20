import numpy as np
import matplotlib.pyplot as plt

a=1.0 #convective velocity
nx=81 #nuber of grid points
dx=2/(nx-1) #grid size
dt=0.01 #time step size
nt=100 #number of time steps
sigma=a*dt/dx #CFL库朗 number, judge the stability of the scheme

x=np.linspace(0,2,nx) #grid points
u=np.exp(-100*(x-1)**2) #Gaussian pulse, the object being transmitted 一开始，在位置x = 1附近有一个 “集中的物理量团”（比如高浓度的流体、局部的温度峰值）。

# MacCormack time stepping
for n in range(nt): #0-99
    u_pred=np.copy(u) #copy
    #predictor step
    u_pred[:-1]=u[:-1]-sigma*(u[1:]-u[:-1])
    #corrector step
    u[1:]=0.5*(u[1:]+u_pred[1:]-sigma*(u_pred[1:]-u_pred[:-1])) #这里的u[1:]，因为u[0]没有前一个点

plt.plot(x,u) #u-x figure
type(x)
type(u)
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Linear Advection — MacCormack Scheme')
plt.show()
