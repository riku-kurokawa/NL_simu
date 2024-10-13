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
a = (C * dt / dx) ** 2
T = int(50*(waveL*dx/C)/dt)*dt  # 計算する時間の長さ
# シミュレーションオブジェクトの作成
nx = int(L / dx) + 1
M = int(T / dt) + 1

frame_interval = int(waveL*dx/C/dt)

u = cp.zeros((M, nx+1))
fix_field = np.zeros((int(M/frame_interval)+1, waveL +100))
x = cp.linspace(0, L, nx+1)
t = cp.linspace(0, T, M)

u[0,50:(50 + waveL)] = -1 * cp.sin(2 * cp.pi * cp.arange(waveL) / waveL)

u[0,0] = 0
u[0,nx] = 0

for i in range(1, nx):
    u[1,i] = u[0,i] + dt * 0 + (a / 2) * (u[0,i + 1] - 2 * u[0,i] + u[0,i - 1])
    
u[1,0] = 0
u[1,nx] = 0

for n in tqdm(range(1, M-1)):
    u[n+1,1:nx] = 2 * u[n,1:nx] - u[n-1,1:nx] + a * (u[n,2:nx+1] - 2 * u[n,1:nx] + u[n,0:nx-1])
    u[n+1,0] = 0
    u[n+1,nx] = 0
    if (n % frame_interval == 0 ):
        fix_field[int(n / frame_interval),:] = u[n+1, int(dt*n*C/dx):100 + waveL + int(dt*n*C/dx)].get()

def animate1d(p, frame_interval, dpi, file_name):
        fig, ax = plt.subplots()
        min_value = np.min(p)
        max_value = np.max(p)
        ax.set_xlim(0, p.shape[1] - 1)
        ax.set_ylim(min_value*2, max_value*2)

        line, = ax.plot(p[0, :], color='blue')

        def update(frame):
            line.set_ydata(p[frame, :])
            ax.set_title(f'Frame {frame}')
            return line,

        ani = FuncAnimation(fig, update, frames=np.arange(0, p.shape[0], frame_interval), interval=200, repeat=True)

        output_file = file_name + '_1D.gif'
        ani.save(output_file, writer='pillow', dpi=dpi)

animate1d(fix_field[:,:] , 1, dpi, file_name)