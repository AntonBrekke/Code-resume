import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


func = input('Input a function: ')
n = int(input('Input number of sum: '))
# Bare formatterings-greier
funclabel = func.replace('**', '^').replace('np.', '').replace('*', '')
if 'exp' in funclabel:
    labellist = funclabel.split(')')
    for i, text in enumerate(labellist):
        if 'exp' in text:
            t = text.replace('exp(', 'e^{') + '}'
            labellist[i] = t
        elif '(' in text and ')' not in text:
            t = text + ')'
            labellist[i] = t
    funclabel = ''.join(labellist)

funclabel = f'${funclabel}$'        # Formatterer penere
interval = input('Give basic interval: ')

a, b = eval(interval)
P = (b - a)/2       # Periodicity-interval ([-L, L] -> P = L)
X = np.linspace(a, b, 10000)
dX = X[1] - X[0]
N = np.arange(1, n+1, 1)

n, x = np.meshgrid(N, X, indexing='ij')

f = lambda x: eval(func)
f1 = eval(func.replace('exp', '~').replace('x', 'X').replace('~', 'exp'))

a0 = 1 / P * np.sum(f(x), axis=1)*dX               # Regner ut Fourier-koeffisienter med wack integral
an = 1 / P * np.sum(f(x) * np.cos(n*np.pi*x/P), axis=1)*dX
bn = 1 / P * np.sum(f(x) * np.sin(n*np.pi*x/P), axis=1)*dX

an = an[:, None]        # Må ha ekstra akser på disse for broadcasting
bn = bn[:, None]        # Må ha ekstra akser på disse for broadcasting

g = an * np.cos(n*np.pi*x/P) + bn * np.sin(n*np.pi*x/P)
fig = plt.figure()
ax = fig.add_subplot()
functext = ax.set_title(f'f(x)={funclabel}', fontsize=16, weight='bold', ha='center')

pmax = np.max(f1)       # Standard hvor-skal-jeg-plassere-teksten-generelt-tingtang
pmin = np.min(f1)       # Standard hvor-skal-jeg-plassere-teksten-generelt-tingtang

# Animerer serie
def update(frame):
    h = a0[frame]/2 + np.sum(g[0:frame, :], axis=0)
    line, = ax.plot(X, f1, 'k')    # tha real func
    line2, = ax.plot(X, h, 'r')      # Fourier-crap
    text = ax.text(b, pmin, f'n={N[frame]}', fontsize=16, weight='bold', ha='right')    # ha gir referansepunkt til teksten
    return line, line2, text,

ani = FuncAnimation(fig, update, frames=len(N), interval=1000, blit=True, repeat=True, repeat_delay=5)
plt.show()
