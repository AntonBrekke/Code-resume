import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt 

"""
Paralellizing "MCI_multivar.py" to get
a distribution of multiple values for 
integral. 
"""

# Start points, end points of integrals
a = [0, 0, 0]     
b = [np.pi, np.pi, np.pi]  
# Number of points in arrays
ni = int(1e3)        
N = 3*[ni]      

# Collection of arrays for each variable 
x = []
for i in range(len(a)):
    x.append(np.linspace(a[i], b[i], N[i]))


def monte_carlo(func, args):
    V = 1
    randargs = []
    f = func
    for xi in args:
        xi0 = xi[0]
        xi1 = xi[-1]
        Nxi = len(xi)
        randxi = np.random.uniform(xi0, xi1, Nxi)
        V *= (xi1 - xi0)
        randargs.append(randxi)
    W = f(randargs)
    MCI = V * np.mean(W)          # Uniform so that probability = area_graph / voluma
    return MCI

# Any arbitrary function, just made a choice
def f(args):
    s = 1
    for xx in args:
        s *= np.sin(xx)
    return s        # Know answer of integral should be 2**n

# Number of MC-cycles
M = int(1e5)
# Configuration list preparing args for function in paralellization
config = M*[[f,x]]

# Paralellization of code
if __name__ == '__main__':
    # Running on 6 threads repeating M times
    proc_num = 6
    # Timing runtime
    t0 = time.time()
    with mp.Pool(proc_num) as pool:
        mc = pool.starmap(monte_carlo, config)      
    t1 = time.time()

    mc = np.array(mc)
    print(f'Computation time {t1 - t0}s')
    print(f'Mean I over all cycles: {np.mean(mc)}')
    print(f'Analytic: {8}')

    # Making figure of data
    fig = plt.figure(facecolor='k')
    ax = fig.add_subplot(facecolor='k')
    ax.spines['bottom'].set_color('w')      # Setting axis white
    ax.spines['left'].set_color('w')
    ax.tick_params(axis='x', colors='w')    # Setting ticks white
    ax.tick_params(axis='y', colors='w')

    # Getting color-chart for histogram
    get_cmap = plt.get_cmap('jet')
    n, bins, patches = ax.hist(mc, bins=75)
    cmap = np.array(get_cmap(n / np.max(n)))    # Must normalize data

    # Coloring patches based on number in bin
    for color, p in zip(cmap, patches):
        plt.setp(p, 'facecolor', color)

    plt.show()