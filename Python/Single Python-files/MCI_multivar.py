import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt 

"""
Generalize "MCI.py" to an arbitrary 
dimensional integral automatically. 
"""

# Start points, end points of integrals
a = [0, 0, 0]     
b = [np.pi, np.pi, np.pi]  
# Number of points in arrays
ni = int(1e6)        
N = 3*[ni]      

# Collection of arrays for each variable 
x = []
for i in range(len(a)):
    x.append(np.linspace(a[i], b[i], N[i]))


def monte_carlo(func, args):
    V = 1
    randargs = []
    for xi in args:
        xi0 = xi[0]
        xi1 = xi[-1]
        Nxi = len(xi)
        randxi = np.random.uniform(xi0, xi1, Nxi)
        V *= (xi1 - xi0)
        randargs.append(randxi)
    W = func(randargs)
    MCI = V * np.mean(W)          
    return MCI


# Any arbitrary function, just made a choice
def f(args):
    s = 1
    for xx in args:
        s *= np.sin(xx)
    return s        

# Timing runtime
t0 = time.time()
I = monte_carlo(f, x)
t1 = time.time()

print(f'Computation time {t1 - t0}s')
print(f'MC: {I}')
# Know answer of integral should be 2**len(x)
print(f'Analytic: {2**len(x)}')


