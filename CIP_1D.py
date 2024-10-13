from time  import *
from pylab import *
from numpy import *
#from scipy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

X = 201
NT = 501

f_p    = zeros((X), "float64")
f_m    = zeros((X), "float64")
g_p    = zeros((X), "float64")
g_m    = zeros((X), "float64")
fn_p   = zeros((X), "float64")
fn_m   = zeros((X), "float64")
gn_p   = zeros((X), "float64")
gn_m   = zeros((X), "float64")

P      = zeros((X), "float64")
dx_P   = zeros((X), "float64")

Ux     = zeros((X), "float64")
dx_Ux  = zeros((X), "float64")


xc= (X-1) // 2


dx = 5.e-2
dt = 5.e-5

Ro = 1.21
bm = 1.4235529e5
c0 = sqrt(bm / Ro)
Z0 = sqrt(bm * Ro)
sigma = 0.2

mic = zeros((NT,X), "float64")

coeff  = zeros(32, "float64")

Ua = c0
xi =-Ua * dt
C  = c0 * dt / dx
C2 = C  * C
C3 = C2 * C

coeff[0]  = (-2. * C3 + 3. * C2)
coeff[1]  = (2. * C3 - 3. * C2 + 1.)
coeff[2]  = xi * (C2 - C)
coeff[3]  = xi * (C2 - 2. * C + 1.)
coeff[4]  = 6. * (-C3 + C2) / xi
coeff[5]  = 6. * (C3 - C2) / xi
coeff[6]  = (3. * C2 - 2. * C)
coeff[7]  = (3. * C2 - 4. * C + 1.)

Ua =-c0
xi =-Ua * dt
C  = c0 * dt / dx
C2 = C  * C
C3 = C2 * C

coeff[8]  = (-2. * C3 + 3. * C2)
coeff[9]  = (2. * C3 - 3. * C2 + 1.)
coeff[10] = xi * (C2 - C)
coeff[11] = xi * (C2 - 2. * C + 1.)
coeff[12] = 6. * (-C3 + C2) / xi
coeff[13] = 6. * (C3 - C2) / xi
coeff[14] = (3. * C2 - 2. * C)
coeff[15] = (3. * C2 - 4. * C + 1.)


for i in range(1, X-1):
    x = dx * i

    TX = x - xc * dx
    P[i]    = \
                    exp(((-TX * TX)) / (2. * sigma**2))
    dx_P[i]  = -TX * \
                    exp(((-TX * TX)) / (2. * sigma**2)) / sigma**2
       


def CIP(coeff0, coeff1, coeff2, coeff3, f0, f1, g0, g1):
    return    coeff0 * f0 \
            + coeff1 * f1 \
            + coeff2 * g0 \
            + coeff3 * g1

start = time.time()

for t in range(NT):

    mic[t,:] = P[:]

    f_p[1:X-1]    =    P[1:X-1] + (Z0 * Ux[1:X-1])
    f_m[1:X-1]    =    P[1:X-1] - (Z0 *    Ux[1:X-1])
    g_p[1:X-1]    = dx_P[1:X-1] + (Z0 * dx_Ux[1:X-1])
    g_m[1:X-1]    = dx_P[1:X-1] - (Z0 * dx_Ux[1:X-1])

    fn_p[1:X-1]   = CIP(coeff[0],  coeff[1],  coeff[2],  coeff[3], \
                                f_p[0:X-2], f_p[1:X-1], \
                                g_p[0:X-2], g_p[1:X-1])
    gn_p[1:X-1]   = CIP(coeff[4],  coeff[5],  coeff[6],  coeff[7], \
                                f_p[0:X-2], f_p[1:X-1], \
                                g_p[0:X-2], g_p[1:X-1])
    fn_m[1:X-1]   = CIP(coeff[8],  coeff[9],  coeff[10], coeff[11], \
                                f_m[2:X], f_m[1:X-1], \
                                g_m[2:X], g_m[1:X-1])
    gn_m[1:X-1]   = CIP(coeff[12], coeff[13], coeff[14], coeff[15], \
                                f_m[2:X], f_m[1:X-1], \
                                g_m[2:X], g_m[1:X-1])

    P[1:X-1]      = (fn_p[1:X-1] + fn_m[1:X-1]) /  2.
    Ux[1:X-1]     = (fn_p[1:X-1] - fn_m[1:X-1]) / (2. * Z0)
    dx_P[1:X-1]   = (gn_p[1:X-1] + gn_m[1:X-1]) /  2.
    dx_Ux[1:X-1]  = (gn_p[1:X-1] - gn_m[1:X-1]) / (2. * Z0)

# アニメーションの設定
x = np.linspace(0, 201, X)

fig, ax = plt.subplots()
line, = ax.plot(x, mic[0, :], color='blue')

# 描画範囲の設定
ax.set_xlim(0, X)  # x軸の表示範囲を0からLに設定
ax.set_ylim(np.min(mic), np.max(mic))  # y軸の表示範囲をuの最小値から最大値に設定

def update(frame):
    line.set_ydata(mic[frame, :])
    return line,


ani = FuncAnimation(fig, update, frames=np.arange(0, NT, 1), interval=10, blit=True)

# アニメーションを保存
output_file = 'animationTest11.gif'
ani.save(output_file, writer='pillow', dpi=100)

# アニメーションの表示
plt.xlabel('x')
plt.ylabel('Displacement')
plt.title('Wave Equation: Displacement vs. x')
plt.show()