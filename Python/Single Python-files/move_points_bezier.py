import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

"""
Making a Bezier-curve given an arbitrary amount of points.
Using matplotlib to make interactive points.
"""

class MoveablePoint:

    all_objects = []
    def __init__(self, ax, graf, obj):
        self.ax = ax
        self.figcanvas = self.ax.figure.canvas
        self.graf = graf
        self.grafB, = ax.plot(0,0)
        self.obj = obj
        self.moved = np.array([obj[0], obj[1]])
        self.point = np.array([obj[0], obj[1]])
        self.pressed = False
        self.start = False
        MoveablePoint.all_objects.append(self)
        # print(MoveablePoint.all_objects)

        self.figcanvas.mpl_connect('button_press_event', self.mouse_press)
        self.figcanvas.mpl_connect('button_release_event', self.mouse_release)
        self.figcanvas.mpl_connect('motion_notify_event', self.mouse_move)

    def mouse_release(self, event):
        if self.ax.get_navigate_mode() != None:
            return
        if not event.inaxes:
            return
        if event.inaxes != self.ax:
            return
        if self.pressed:
            self.start = False
            self.obj = self.moved
            self.pressed = False
            self.point = np.array([event.xdata, event.ydata])
            return

    def mouse_press(self, event):
        if self.ax.get_navigate_mode() != None:
            return
        if not event.inaxes:
            return
        if event.inaxes != self.ax:
            return
        if self.start:
            return
        if event.xdata >= self.point[0] - 0.15 and event.xdata <= self.point[0] + 0.15 and event.ydata >= self.point[1] - 0.15 and event.ydata <= self.point[1] + 0.15:
            self.pressed = True
            self.point = np.array([event.xdata, event.ydata])

    def mouse_move(self, event):
        if self.ax.get_navigate_mode() != None:
            return
        if not event.inaxes:
            return
        if event.inaxes != self.ax:
            return
        if not self.pressed:
            return
        self.start = True

        self.x = self.point[0] - event.xdata
        self.y = self.point[1] - event.ydata
        self.moved = self.obj - [self.x, self.y]
        mvdx, mvdy = self.moved[0], self.moved[1]

        # print('on da move')

        print(self.moved)
        self.graf.remove()
        self.graf, = self.ax.plot(mvdx, mvdy, color='r', marker='o', markersize=8, zorder=3)
        self.grafB, = self.BezierCurve()

        self.figcanvas.draw()
        self.grafB.remove()

    def BezierCurve(self):
        t = np.linspace(0, 1, 300)[:,None]
        bezier_points = MoveablePoint.all_objects
        n = len(bezier_points)
        # print(n)
        B = 0
        for i in range(n):
            B += ss.binom(n-1, i)*(1-t)**((n-1)-i)*t**i*bezier_points[i].moved
            # print(ss.binom(n-1,i))        # n-1 fordi n=4 pga. len(), men n skal være n=3
        return self.ax.plot(B[:,0], B[:,1], 'k')


fig, ax1 = plt.subplots()
ax1.set_ylim(-10, 10)
ax1.set_xlim(-10, 10)
ax1.axis('off')

def num_points(num):
    np.random.seed(0)
    nums = np.random.uniform(-9, 9, (num, 2))
    for i, j in zip(nums[:,0], nums[:,1]):
        P = np.array([i, j])
        gmdl, = ax1.plot(P[0], P[1], color='r', marker='o', markersize=8, zorder=3)
        moviepoint = MoveablePoint(ax1, gmdl, P)

n_points = input('Enter number of points: ')
num_points(int(n_points))
plt.show()
