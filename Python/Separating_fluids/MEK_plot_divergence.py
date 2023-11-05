from MEK_data_class import *

# Separating data
fluid_array = method.data_sep()[:,0]
gas_array = method.data_sep()[:,1]

y_fluid, l_fluid, u_fluid, v_fluid = fluid_array
y_gas, l_gas, u_gas, v_gas = gas_array

# Spacing in xy-grid is 0.5
div_curl_gas = div_curl(u_gas, v_gas, 0.5, 0.5)
div_curl_fluid = div_curl(u_fluid, v_fluid, 0.5, 0.5)

# Getting divergence
div_gas = div_curl_gas.divergence()
div_fluid = div_curl_fluid.divergence()

# Plotting divergence
fig = plt.figure()
ax = fig.add_subplot(111)

divergence_gas = ax.contourf(x, y_gas, div_gas, 200, cmap='hsv')
divergence_fluid = ax.contourf(x, y_fluid, div_fluid, 200, cmap='hsv')
ax.plot(xit[0], yit[0], color='k', linewidth=6, label='Separation line')
cbar_div_gas = fig.colorbar(divergence_gas, ax=ax)
cbar_div_gas.set_label(r'Divergence gas [1/s]', weight='bold')
cbar_div_fluid = fig.colorbar(divergence_fluid, ax=ax)
cbar_div_fluid.set_label(r'Divergence fluid [1/s]', weight='bold')

ax.set_xlabel(r'x [mm]', weight='bold')
ax.set_ylabel(r'y [mm]', rotation=0, weight='bold', position=(0,0.52))

corners = [(35,160,70,170),(35,85,70,100),(35,50,70,60)]
method.plot_rectangles(ax, corners)

ax.legend(bbox_to_anchor=(1.017,1.1))
fig.tight_layout()
plt.show()

# Kjøretest fra terminal:
"""
PS C:> python .\klasser_b.py
PS C:>
"""
