import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

"""
Code that illustrates convolution.
Classes is mostly for fun, making 
function-types that can be added, subtracted,
multiplied, or divided before evaluation. 

Second part illustrate the convolution between two
chosen functions constructed from this class. 
"""

class interactive_func:
    """
    Class that makes it easily possible to have operations
    between different functions.
    """
    def __init__(self, func):
        self.func = func

    def __add__(self, other):
        def summed(*args, **kwargs):
            return self(*args, **kwargs) + other(*args, **kwargs)
        return interactive_func(summed)

    def __sub__(self, other):
        def subtracted(*args, **kwargs):
            return self(*args, **kwargs) - other(*args, **kwargs)
        return interactive_func(subtracted)

    def __mul__(self, other):
        def multiplied(*args, **kwargs):
            return self(*args, **kwargs) * other(*args, **kwargs)
        return interactive_func(multiplied)

    def __truediv__(self, other):
        def divided(*args, **kwargs):
            return self(*args, **kwargs) / other(*args, **kwargs)
        return interactive_func(divided)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# Define different functions as subclasses
class unit_square(interactive_func):
    def __init__(self, a, height=1):
        self.a = a
        self.h = height
        self.w = 1 / height

    def __call__(self, x):
        a = self.a
        h = self.h
        w = self.w
        return h * (x > a - w/2) * (x < a + w/2)

class gaussian_wavelet(interactive_func):
    def __init__(self, a, omega=0, A=1, sigma=1):
        self.a = a
        self.omega = omega
        self.A = A
        self.sigma = sigma

    def __call__(self, x):
        a = self.a; sigma = self.sigma
        A = self.A; omega = self.omega
        return A*np.exp(-((x-a)/sigma)**2)*np.cos(omega*x)

class stepfunc(interactive_func):
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        a = self.a
        return np.ones_like(x) * (x >=a)

class triangle(interactive_func):
    def __init__(self, a, width=1, height=1):
        self.a = a
        self.width = width
        self.height = height

    def __call__(self, x):
        a = self.a
        h = self.height
        w = self.width/2        # Width of triangle on each side of the center
        return h/w*((x-(a-w))*(x>a-w)*(x<a) + (-(x-a)+w) * (x>=a) * (x<a+w))


def convolution(x, f, g):
    """
    Function that calculates the convolution integral fill_between
    two functions
    """
    tau = x.copy()      # Only for readability
    dtau = tau[1] - tau[0]
    s = np.zeros_like(x)
    for i, t in enumerate(x):
        s[i] = np.sum(f(tau)*g(t - tau)) * dtau
    return s

N = 10000
x = np.linspace(-6, 6, N)
dx = x[1] - x[0]

# First function
s1 = triangle(0)
for i in range(1, 10):
    s1 += triangle(i, height=1/(i+1)) + triangle(-i, height=1/(i+1))

# Second function
s2 = unit_square(0)

numpy_conv = np.convolve(s1(x), s2(x), mode='same') * dx

# Making plots
plt.style.use('seaborn-v0_8-ticks')

plt.plot(x, s1(x), 'tab:blue', label='s1')
plt.plot(x, s2(x), 'k', label='s2')
plt.plot(x, numpy_conv, 'tab:green', label='Numpy convolution')

conv = convolution(x, s1, s2)
plt.plot(x, conv, 'tab:red', label='Homemade convolution', linestyle='--')

plt.legend()
plt.show()

# Making figure to animate 
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(x, np.zeros_like(x), 'gray', linestyle='--')
ax2.plot(x, np.zeros_like(x), 'gray', linestyle='--')

plot1, = ax1.plot(x, s1(x), 'tab:blue', label='s1')
plot2, = ax1.plot(x, s2(x), 'k', label='s2')
plot3, = ax2.plot(x, conv, 'tab:red', linewidth=2, label='convolution')
ax1.legend()
ax2.legend()

# Want to animate convolution
def update(frame):
    a = x[frame]
    plot2.set_data(x, s2(a-x))
    plot3.set_data(x[:frame], conv[:frame])

    return plot2, plot3

ani = FuncAnimation(fig, update, frames=range(0,N,int(N/500)), interval=1, blit=True)
fig.tight_layout()
plt.show()
# ani.save('convolution.gif', writer=PillowWriter(fps=25))
