import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
import sympy as sy
from matplotlib.animation import FuncAnimation
import sympy.physics.mechanics as spm
from matplotlib import collections  as mc

"""
l2 = 15 and l4 = 12
l2 = 6 and l4 = 6 (best yet)
"""

# Defining constants of the system
m1 = 7000
m2 = 15
l1 = 1.5
l2 = 10          # l2 = 6 is even better!
l3 = 2
l4 = 6          # l4 = 8 is really good!
g = 9.81
theta_i = 5*np.pi/6
phi_i = 0
psi_i = 0

Nt = int(1e6)
t0 = 0
t1 = 2
t = np.linspace(t0, t1, Nt)

Lmax = 2*m1/m2*l1*(1 - np.cos(theta_i))
"""
Let
x1 = theta, x2 = dtheta/dt
x3 = phi, x4 = dphi/dt
x5 = psi, x5 = dpsi/dt

x2d = thetadd, x4d = phidd, x6d = psidd
"""

# q1 = theta, q2 = phi, q3 = psi

q1, q2, q3 = spm.dynamicsymbols('q1 q2 q3')
q1d, q2d, q3d = spm.dynamicsymbols('q1 q2 q3', 1)       # First order derivative of q1, q2, q3
q1dd, q2dd, q3dd = spm.dynamicsymbols('q1 q2 q3', 2)       # Second order derivative of q1, q2, q3

# Implement Lagrangian of the system and get the equations of motion (this is only time I can fuck up I think)
K = 1/2*m1*(l1**2*q1d**2 + l3**2*q2d**2 + 2*l1*l3*q1d*q2d*sy.cos(q1 - q2)) + 1/2*m2*(l2**2*q1d**2 + l4**2*q3d**2 + 2*l2*l4*q1d*q3d*sy.sin(q1 + q3))
V = m1*g*(-l1*sy.cos(q1) - l3*sy.cos(q2)) + m2*g*(l2*sy.cos(q1) - l4*sy.sin(q3))

L = K - V

LM = spm.LagrangesMethod(L, [q1, q2, q3])
eom_theta, eom_phi, eom_psi = LM.form_lagranges_equations()

sol = sy.solve([eom_theta, eom_phi, eom_psi], [q1dd, q2dd, q3dd])

x2d_sol = sol[q1dd]
x4d_sol = sol[q2dd]
x6d_sol = sol[q3dd]

x2d_func = sy.lambdify([q1, q1d, q2, q2d, q3, q3d], x2d_sol)
x4d_func = sy.lambdify([q1, q1d, q2, q2d, q3, q3d], x4d_sol)
x6d_func = sy.lambdify([q1, q1d, q2, q2d, q3, q3d], x6d_sol)

# Solving ODE

def Lagrange_equations(t, Y):
    """
    Solving Lagranges equations in this set. Let
    Y[0] = x1 = theta, Y[1] = x2 = dtheta/dt,
    Y[2] = x3 = phi, Y[3] = x4 = dphi/dt
    Y[4] = x5 = psi, Y[5] = x6 = dpsi/dt
    """
    dY = np.zeros_like(Y)
    x1, x2, x3, x4, x5, x6 = Y
    dY[1] = x2d_func(x1, x2, x3, x4, x5, x6)
    dY[0] = x2
    dY[3] = x4d_func(x1, x2, x3, x4, x5, x6)
    dY[2] = x4
    dY[5] = x6d_func(x1, x2, x3, x4, x5, x6)
    dY[4] = x6
    return dY

Y0 = np.array([theta_i, 0, phi_i, 0, psi_i, 0])

sol = scint.solve_ivp(Lagrange_equations, [t0, t1], Y0, method='Radau', dense_output=True, rtol=1e-10, atol=1e-10)

Y = sol.sol(t)
theta, dthetadt, phi, dphidt, psi, dpsidt = Y

x1 = l1*np.sin(theta) + l3*np.sin(phi)   # Counterweight
y1 = -l1*np.cos(theta) - l3*np.cos(phi)
x2 = -l2*np.sin(theta) + l4*np.cos(psi)      # Cow
y2 = l2*np.cos(theta) - l4*np.sin(psi)

x1d = l1*dthetadt*np.cos(theta) + l3*dphidt*np.cos(phi)
y1d = l1*dthetadt*np.sin(theta) + l3*dphidt*np.sin(phi)
x2d = -l2*dthetadt*np.cos(theta) - l4*dpsidt*np.sin(psi)
y2d = -l2*dthetadt*np.sin(theta) - l4*dpsidt*np.cos(psi)

v2 = np.sqrt(x2d**2 + y2d**2)

alpha = np.arctan2(y2d, x2d)

L = v2**2*np.sin(2*alpha) / g

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot()

"""
P = np.linspace(-1e5, 1e5, 100)
T = np.linspace(-2*np.pi, 2*np.pi, 100)

theta, p = np.meshgrid(T, P, indexing='xy')

dthetadt = p / (m1*l1**2 + m2*l2**2)
dpdt = -(m1*l1 - m2*l2)*g*np.sin(theta)

ax1.streamplot(theta, p, dthetadt, dpdt)
"""

eff = L/Lmax
max_eff = np.max(eff)
max_index = np.where(eff == max_eff)
ax.plot(theta, eff, label=r'ϵ$_L$')
ax.plot(np.ones(10)*np.pi/4, np.linspace(np.min(eff), np.max(eff), 10), 'k', label=r'$\pi$/4')
ax.plot(np.ones(10)*theta[max_index], np.linspace(np.min(eff), np.max(eff), 10), 'r--', label=r'θ$_{f,max}$')

ax.set_xlabel(r'θ$_f$ [rad]', fontsize=16)
ax.set_ylabel(r'ϵ$_L$ [-]', fontsize=16)

ax.legend(prop={'size':16})
fig.tight_layout()
plt.show()

phi_compare = phi[max_index] - theta[max_index] + np.pi
psi_compare = psi[max_index] + theta[max_index] - np.pi/2
print(f'max_eff: {max_eff}, L/Lmax: {np.max(L):.2f}/{Lmax:.2f}\nalpha_max: {alpha[max_index]},{alpha[max_index]*180/np.pi}\ntheta_max: {theta[max_index]},{theta[max_index]*180/np.pi}\nphi_max: {phi[max_index]},{phi[max_index]*180/np.pi}\npsi_max: {psi[max_index]},{psi[max_index]*180/np.pi}\nphi_compare: {phi_compare*180/np.pi}\npsi_compare: {psi_compare*180/np.pi}\nt: {t[max_index]}s')


fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot()
m1_plot, = ax.plot([], [], 'r', marker='o')
m2_plot, = ax.plot([], [], 'r', marker='o')
line1a, = ax.plot(x1, y1, 'tab:blue')
line1b, = ax.plot(x1, y1, 'tab:blue')
line2a, = ax.plot(x2, y2, 'tab:blue')
line2b, = ax.plot(x2, y2, 'tab:blue')
ax.plot(0,0,'ko')

anim_slice = int(8e3 / t1)     # For animation purposes (random ratio that fits well for most t1)
x1_anim = x1.copy()[::anim_slice]
y1_anim = y1.copy()[::anim_slice]
x2_anim = x2.copy()[::anim_slice]
y2_anim = y2.copy()[::anim_slice]
t_anim = t.copy()[::anim_slice]

trace = [np.column_stack([[x2_anim[i], x2_anim[i+1]], [y2_anim[i], y2_anim[i+1]]]) for i in range(0, len(x2_anim)-1)]    # Want to trace ball nr. 2
lc = mc.LineCollection(trace, cmap='jet', linewidths=2, zorder=-1)      # Want it to appear beneath everything, zorder=0.
lc.set_array(t_anim)     # Trace with respect to time
ax.add_collection(lc)

colorbar = fig.colorbar(lc, ax=ax)
colorbar.set_label('Time [s]', color='k', fontsize=16, weight='bold')

ax.axis('equal')
ax.grid(True)

ax.set_xlabel('x [m]', fontsize=16)
ax.set_ylabel('y [m]', fontsize=16)

x1a = l1*np.sin(theta[::anim_slice])
y1a = -l1*np.cos(theta[::anim_slice])
x2a = -l2*np.sin(theta[::anim_slice])
y2a = l2*np.cos(theta[::anim_slice])

time = ax.text(10, -5, f't={t[-1]:.2f}s', color='k', ha='center', va='top')
def update(frame):
    m1_plot.set_data(x1_anim[frame], y1_anim[frame])
    m2_plot.set_data(x2_anim[frame], y2_anim[frame])
    line1a.set_data([0, x1a[frame]], [0, y1a[frame]])
    line1b.set_data([x1a[frame], x1_anim[frame]], [y1a[frame], y1_anim[frame]])
    line2a.set_data([0, x2a[frame]], [0, y2a[frame]])
    line2b.set_data([x2a[frame], x2_anim[frame]], [y2a[frame], y2_anim[frame]])
    time.set_text(f't = {t_anim[frame]:.2f}s')
    lc.set_segments(trace[:frame])
    return lc, m1_plot, m2_plot, line1a, line1b, line2a, line2b, time

anim_speed = 1
ani = FuncAnimation(fig, update, frames=[i for i in range(0, len(t_anim), anim_speed)],
                    blit=True, interval=5, repeat=True)
plt.show()

# Plotting energy to check if it is constant
K = 1/2*m1*(x1d**2 + y1d**2) + 1/2*m2*(x2d**2 + y2d**2)
V = m1*g*y1 + m2*g*y2
E = K + V

fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(t, V, 'r', label='Potential energy, V')
ax1.plot(t, K, label='Kinetic energy, K')
# ax.set_ylim(np.min(E) - 10, np.max(E) + 10)       # To see whats going on
ax1.plot(t, E, color='purple', label='Total energy, E=K+V')
ax2.plot(t, E, color='purple', label='Total energy, E=K+V')

ax1.set_xlabel('t [s]', fontsize=16)
ax1.set_ylabel('E [J]', fontsize=16)
ax2.set_xlabel('t [s]', fontsize=16)
ax2.set_ylabel('E [J]', fontsize=16)

ax1.legend(prop={'size':16}, loc='upper left')
ax2.legend(prop={'size':16}, loc='upper right')

fig.tight_layout()
plt.show()
