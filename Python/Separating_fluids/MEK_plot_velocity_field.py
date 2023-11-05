from MEK_data_class import *

# Separating data
fluid_array = method.data_sep()[:,0]
gas_array = method.data_sep()[:,1]

y_fluid, l_fluid, u_fluid, v_fluid = fluid_array
y_gas, l_gas, u_gas, v_gas = gas_array

# u_gas = u_gas / l_gas
# v_gas = v_gas / l_gas
# u_fluid = u_fluid / l_fluid
# v_fluid = v_fluid / l_fluid

# Plotting velocity-fields
corners = [(35,160,70,170),(35,85,70,100),(35,50,70,60)]        # Selecting some regions
N = 8
fig = plt.figure()
ax = fig.add_subplot(111)
v_field_gas = ax.quiver(x[::N, ::N], y_gas[::N, ::N], u_gas[::N, ::N], v_gas[::N, ::N], l_gas[::N, ::N], cmap='jet')
v_field_fluid = ax.quiver(x[::N, ::N], y_fluid[::N, ::N], u_fluid[::N, ::N], v_fluid[::N, ::N], l_fluid[::N, ::N], cmap='jet')
ax.plot(xit[0], yit[0], color='k', linewidth=3, label='Separation line')
quiver_cbar_gas = fig.colorbar(v_field_gas, ax=ax)
quiver_cbar_gas.set_label('Speed gas [mm/s]', weight='bold')
quiver_cbar_fluid = fig.colorbar(v_field_fluid, ax=ax)
quiver_cbar_fluid.set_label('Speed fluid [mm/s]', weight='bold')
method.plot_rectangles(ax, corners)
ax.set_xlabel(r'x [mm]', weight='bold')
ax.set_ylabel(r'y [mm]', rotation=0, weight='bold', position=(0,0.52))

ax.legend(bbox_to_anchor=(1.017,1.1))
plt.show()

