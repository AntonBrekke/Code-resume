import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
import multiprocessing as mp

a = 0
b = np.pi
N = 25000
M = 100000

t0 = time.time()

@njit
def f(x):
    return np.sin(x)

@njit(cache=True)
def monte_carlo(repeat_dummy):
    randx = np.random.uniform(a, b, N)
    sum = np.sum(f(randx))
    integral = (b-a) / N * sum          # Uniform, so probability = area_graph / area_box
    return integral


if __name__ == '__main__':
    t0 = time.time()

    proc_num = 10
    with mp.Pool(proc_num) as pool:
        mc = pool.map(monte_carlo, np.ones(M))

    mc = np.array(mc)
    t1 = time.time()
    print(f'Computation time {t1 - t0}s')

    # Lager en figur som skal se kulere ut
    fig = plt.figure(facecolor='k')
    ax = fig.add_subplot(facecolor='k')
    ax.spines['bottom'].set_color('w')      # Setter aksen hvit
    ax.spines['left'].set_color('w')
    ax.tick_params(axis='x', colors='w')    # Setter ticks hvit
    ax.tick_params(axis='y', colors='w')

    # Henter fargekart for histogram
    get_cmap = plt.get_cmap('jet')
    n, bins, patches = ax.hist(mc, bins=75)
    cmap = np.array(get_cmap(n / np.max(n)))    # Må normalisere data

    for color, p in zip(cmap, patches):
        plt.setp(p, 'facecolor', color)

    plt.show()
