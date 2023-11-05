import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
import multiprocessing as mp

"""
Really simple example of Monte-Carlo integral
with paralellized code.
"""

a = 0
b = np.pi
N = 25000       # Number of points
M = 100000      # Number of Monte-Carlo cycles

t0 = time.time()

@njit
def f(x):
    return np.sin(x)        # True integral value of this on 0, pi is 2 

@njit(cache=True)
def monte_carlo(repeat_dummy):
    randx = np.random.uniform(a, b, N)
    sum = np.sum(f(randx))
    integral = (b-a) / N * sum          # Uniform, so probability = area_graph / area_box
    return integral


if __name__ == '__main__':
    t0 = time.time()

    proc_num = 6
    with mp.Pool(proc_num) as pool:
        mc = pool.map(monte_carlo, np.ones(M))      # Running on 6 threads repeating M times

    mc = np.array(mc)
    t1 = time.time()
    print(f'Computation time {t1 - t0}s')

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

    for color, p in zip(cmap, patches):
        plt.setp(p, 'facecolor', color)

    plt.show()
