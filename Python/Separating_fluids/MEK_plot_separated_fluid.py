from MEK_data_class import *

# Separating data
fluid_array = method.data_sep()[:,0]
gas_array = method.data_sep()[:,1]

y_fluid, l_fluid, u_fluid, v_fluid = fluid_array
y_gas, l_gas, u_gas, v_gas = gas_array

# Plotting data in one plot 
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(xit[0], yit[0], color='w', linewidth=3, label='Separation line')
ax.set_xlabel(r'x [mm]', weight='bold')
ax.set_ylabel(r'y [mm]', rotation=0, weight='bold', position=(0,0.52))
gas = ax.contourf(x, y_gas, l_gas, 200, cmap='viridis')
fluid = ax.contourf(x, y_fluid, l_fluid, 200, cmap='inferno')
cbar_gas = fig.colorbar(gas, ax=ax)
cbar_fluid = fig.colorbar(fluid, ax=ax)
cbar_gas.set_label(r'Gas speed [mm/s]', weight='bold')
cbar_fluid.set_label(r'Fluid speed [mm/s]', weight='bold')
ax.legend(bbox_to_anchor=(1.017,1))
fig.tight_layout()
plt.show()

# Plotting gas and fluid separately
fig2 = plt.figure()
gs = fig2.add_gridspec(2,1)
ax2 = fig2.add_subplot(gs[1,0])
ax3 = fig2.add_subplot(gs[0,0])

ax2.set_title('Fluid', weight='bold')
ax2.plot(xit[0], yit[0], color='k', linewidth=3, label='Separation line')
ax2.set_xlabel(r'x [mm]', weight='bold')
ax2.set_ylabel(r'y [mm]', rotation=0, weight='bold', position=(0,0.52))
fluid2 = ax2.contourf(x, y_fluid, l_fluid, 200, cmap='jet')
cbar_fluid2 = fig2.colorbar(fluid2, ax=ax2)
cbar_fluid2.set_label(r'Fluid speed [mm/s]', weight='bold')

ax3.set_title('Gas', weight='bold')
ax3.plot(xit[0], yit[0], color='k', linewidth=3, label='Separation line')
ax3.set_xlabel(r'x [mm]', weight='bold')
ax3.set_ylabel(r'y [mm]', rotation=0, weight='bold', position=(0,0.58))
gas3 = ax3.contourf(x, y_gas, l_gas, 200, cmap='jet')
cbar_gas3 = fig2.colorbar(gas3, ax=ax3)
cbar_gas3.set_label(r'Gas speed [mm/s]', weight='bold')
ax2.legend(bbox_to_anchor=(1.017,1.25)); ax3.legend(bbox_to_anchor=(1.017,1.25))
fig2.tight_layout()
plt.show()


