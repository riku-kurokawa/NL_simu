import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import cupy as cp

dpi, file_name = 100, "nl-fdm_1D_v1.0"

L = 0.5 # 空間の長さ
dx = 5.0e-5
dt = 5.0e-8
waveL = int(0.008575 / dx)
C = 343
Ro = 1.21
a =  dt / (2*dx)
b = C*C
d = C**2*Ro
T = int(50*(waveL*dx/C)/dt)*dt  # 計算する時間の長さ
nx = int(L / dx) + 1
M = int(T / dt) + 1
frame_interval = int(waveL*dx/C/dt)

p = cp.zeros((M, nx+1))
u = cp.zeros((M, nx+1))
fix_field = np.zeros((int(M/frame_interval)+1, waveL +100))
x = cp.linspace(0, L, nx+1)
t = cp.linspace(0, T, M)

p[0,50:(50 + waveL)] = -100 * cp.sin(2 * cp.pi * cp.arange(waveL) / waveL)
u[0,50:(50 + waveL)] = -100 *cp.sin(2 * cp.pi * cp.arange(waveL)/ waveL) / Ro /C



for n in tqdm(range(0, M-1)):
    #p[n+1,1:nx] = p[n,1:nx] - a* ((p[n,1:nx] + d) * (u[n,2:nx + 1]-u[n,0:nx - 1]) + u[n,1:nx]*(p[n,2:nx + 1]-p[n,0:nx - 1]))
    u[n+1,1:nx] = u[n,1:nx] - a* (u[n,1:nx]*(u[n,2:nx + 1]-u[n,0:nx - 1]) + (p[n,2:nx + 1]-p[n,0:nx - 1]) / (p[n,1:nx] / b + Ro))
    p[n+1,1:nx] = p[n,1:nx] - a* ((d + p[n,2:nx+1])*(u[n+1,2:nx + 1]) - (d + p[n+1,0:nx-1] )*(u[n+1,0:nx - 1]))
    #u[n+1,1:nx] = u[n,1:nx] - a* (p[n,2:nx + 1]-p[n,0:nx - 1]) / Ro
    #p[n+1,1:nx] = p[n,1:nx] - a* Ro *b*(u[n+1,2:nx + 1] - u[n+1,0:nx - 1])

    u[n+1,0] = 0
    u[n+1,nx] = 0
    p[n+1,0] = 0
    p[n+1,nx] = 0

    if (n % frame_interval == 0 ):
        fix_field[int(n / frame_interval),:] = p[n+1, int(dt*n*C/dx):100 + waveL + int(dt*n*C/dx)].get()

def animate1d(y, frame_interval, dpi, file_name):
        fig, ax = plt.subplots()
        min_value = np.min(y)
        max_value = np.max(y)
        ax.set_xlim(0, y.shape[1] - 1)
        ax.set_ylim(min_value*2, max_value*2)

        line, = ax.plot(y[0, :], color='blue')

        def update(frame):
            line.set_ydata(y[frame, :])
            ax.set_title(f'Frame {frame}')
            return line,

        ani = FuncAnimation(fig, update, frames=np.arange(0, y.shape[0], frame_interval), interval=200, repeat=True)

        output_file = file_name + '_1D.gif'
        ani.save(output_file, writer='pillow', dpi=dpi)

animate1d(fix_field[:,:] , 1, dpi, file_name)