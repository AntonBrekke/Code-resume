import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

"""
Class which separates fluid and gas given a separation-line. 
"""

# Work with version 7 MAT-files 
data = sio.loadmat('data.mat')
x = data.get('x')
y = data.get('y')
u = data.get('u')       # i-component of velocity
v = data.get('v')       # j-component of velocity
xit = data.get('xit')
yit = data.get('yit')

l = np.sqrt(u**2 + v**2)

class data_field:
    def __init__(self, x, y, u, v, xit, yit):
        self.x = x; self.y = y;
        self.u = u; self.v = v
        self.xit = xit; self.yit = yit;
        self.l = np.sqrt(u**2 + v**2)
    # a)
    def shapes(self):
        print(f'shape x = {np.shape(self.x)}')
        print(f'shape y = {np.shape(self.y)}')
        print(f'shape u = {np.shape(self.u)}')
        print(f'shape v = {np.shape(self.v)}')
        print(f'shape xit = {np.shape(self.xit)}')
        print(f'shape yit = {np.shape(self.yit)}')

        print(f'intersections in grid : {len(self.x[0])*len(self.y[:,0])}')

        print(f'all dx = 0.5 : {np.all(np.diff(self.x, axis=1)==0.5)}')
        print(f'all dy = 0.5 : {np.all(np.diff(self.y, axis=0)==0.5)}')

        print(f'first and last y-values: \n{self.y[0,0]} {self.y[0,1]}\n{self.y[-1,0]} {self.y[-1,-1]}')
    # b) - f)
    def data_sep(self):
        self.y_gas = self.y.copy(); self.y_fluid = self.y.copy()
        self.l_gas = self.l.copy(); self.l_fluid = self.l.copy()
        self.u_gas = self.u.copy(); self.v_gas = self.v.copy()
        self.u_fluid = self.u.copy(); self.v_fluid = self.v.copy()
        # Separating data on a given line 
        for m in range(len(self.y[:,0])):
            for n in range(len(self.y[0])):
                if self.y[m,n] >= self.yit[0,n]:
                    self.y_fluid[m,n] = self.yit[0,n]
                    self.l_fluid[m,n] = np.nan
                    self.u_fluid[m,n] = np.nan; self.v_fluid[m,n] = np.nan
                elif self.y[m,n] < self.yit[0,n]:
                    self.y_gas[m,n] = self.yit[0,n]
                    self.l_gas[m,n] = np.nan
                    self.u_gas[m,n] = np.nan; self.v_gas[m,n] = np.nan
        self.data_sep_array = np.array([[self.y_fluid, self.y_gas],
                                        [self.l_fluid, self.l_gas],
                                        [self.u_fluid, self.u_gas],
                                        [self.v_fluid, self.v_gas]])
        return self.data_sep_array

    def plot_rectangles(self, ax, corners):     # corners ex: [(35,160,70,170),(35,85,70,100),(35,50,70,60)]
        for C in corners:
            x1, x2 = C[0]-1, C[2]-1
            y1, y2 = C[1]-1, C[3]-1
            # Rektangler
            ax.plot([x[0, x1], x[0, x2]], [y[y1, 0], y[y1, 0]], 'r')   # Bottom C1
            ax.plot([x[0, x2], x[0, x2]], [y[y1, 0], y[y2, 0]], 'g')   # Right C2
            ax.plot([x[0, x2], x[0, x1]], [y[y2, 0], y[y2, 0]], 'b')   # Top C3
            ax.plot([x[0, x1], x[0, x1]], [y[y2, 0], y[y1, 0]], 'k')   # Left C4

# Subclass of data for divergens, curl, circulation og flux
class div_curl(data_field):
    def __init__(self, u, v, dx, dy):   # v = ui + vj
        data_field.__init__(self, x, y, u, v, xit, yit)
        self.dx = dx; self.dy = dy
    # d)
    def divergence(self):
        self.dudx = np.gradient(self.u, self.dx, axis=1)
        self.dvdy = np.gradient(self.v , self.dy, axis=0)
        return self.dudx + self.dvdy
    # e)
    def curl(self):
        self.dudy = np.gradient(self.u, self.dx, axis=0)
        self.dvdx = np.gradient(self.v, self.dy, axis=1)
        return self.dvdx - self.dudy
    # f)
    def circulation(self, corners):
        dx = self.dx; dy = self.dy
        u = self.u; v = self.v
        for i, C in enumerate(corners, start=1):
            x1, x2 = C[0]-1, C[2]-1
            y1, y2 = C[1]-1, C[3]-1

            u_C1 = u[y1, x1:x2+1]
            u_C3 = u[y2, x1:x2+1]
            v_C2 = v[y1:y2+1, x2]
            v_C4 = v[y1:y2+1, x1]

            circ_tot = np.sum([np.sum(u_C1)*dx,
                               np.sum(v_C2)*dy,
                               np.sum(u_C3)*(-dx),
                               np.sum(v_C4)*(-dy)])

            stokes_curl = self.curl()[y1:y2+1, x1:x2+1]
            stokes_int = dx*dy*(np.sum(np.sum(stokes_curl, axis=1), axis=0))

            print(f'\nRectangle {i}:\n Direct : {circ_tot} \n Stokes : {stokes_int}\n')
            print(f'Each sides : \n C1: {np.sum(u_C1)*dx} (bottom) \n C2: {np.sum(v_C2)*dy} (right) \n C3: {np.sum(u_C3)*(-dx)} (top) \n C4: {np.sum(v_C4)*(-dy)} (left)')
        print('')
    # g)
    def flux_sides(self, corners):
        dx = self.dx; dy = self.dy
        u = self.u; v = self.v
        for i, C in enumerate(corners, start=1):
            x1, x2 = C[0]-1, C[2]-1
            y1, y2 = C[1]-1, C[3]-1

            v_C1 = v[y1, x1:x2+1]
            v_C3 = v[y2, x1:x2+1]
            u_C2 = u[y1:y2+1, x2]
            u_C4 = u[y1:y2+1, x1]

            flux_tot = np.sum([np.sum(-v_C1*dx),
                               np.sum(u_C2)*dy,
                               np.sum(v_C3)*dx,
                               np.sum(-u_C4)*dy])

            print(f'\nFlux Rectangle {i}: {flux_tot}')
            print(f'Each sides : \n C1: {np.sum(-v_C1)*dx} (bottom) \n C2: {np.sum(u_C2)*dy} (right) \n C3: {np.sum(v_C3)*dx} (top) \n C4: {np.sum(-u_C4)*dy} (left)')
        print('')

method = data_field(x, y, u, v, xit, yit)
