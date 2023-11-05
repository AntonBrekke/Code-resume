from MEK_data_class import *

# Separating data
fluid_array = method.data_sep()[:,0]
gas_array = method.data_sep()[:,1]
y_fluid, l_fluid, u_fluid, v_fluid = fluid_array
y_gas, l_gas, u_gas, v_gas = gas_array

# Spacing in xy-grid is 0.5
div_curl_gas = div_curl(u_gas, v_gas, 0.5, 0.5)
div_curl_fluid = div_curl(u_fluid, v_fluid, 0.5, 0.5)
curl_gas = div_curl_gas.curl()
curl_fluid = div_curl_fluid.curl()

# Plotting curl
fig = plt.figure()
ax = fig.add_subplot(111)

curlz_gas = ax.contourf(x, y_gas, curl_gas, 200, cmap='hsv')
curlz_fluid = ax.contourf(x, y_fluid, curl_fluid, 200, cmap='jet')
cbar_curlz_gas = fig.colorbar(curlz_gas, ax=ax)
cbar_curlz_fluid = fig.colorbar(curlz_fluid, ax=ax)
cbar_curlz_gas.set_label(r'Curl gas [1/s]', weight='bold')
cbar_curlz_fluid.set_label(r'Curl fluid [1/s]', weight='bold')
ax.plot(xit[0], yit[0], color='k', linewidth=6, label='Separation line')
ax.set_xlabel(r'x [mm]', weight='bold')
ax.set_ylabel(r'y [mm]', rotation=0, weight='bold', position=(0,0.52))
corners = [(35,160,70,170),(35,85,70,100),(35,50,70,60)]
method.plot_rectangles(ax, corners)
ax.legend(bbox_to_anchor=(1.017,1.1))

fig.tight_layout()
plt.show()

# Making Streamplot
fig = plt.figure()
ax = fig.add_subplot(111)

stream_fluid = ax.streamplot(x, y, u_fluid, v_fluid, color=l_fluid, cmap='jet')
stream_gas = ax.streamplot(x, y, u_gas, v_gas, color=l_gas, cmap='jet')
ax.plot(xit[0], yit[0], color='k', linewidth=3, label='Separation line')
ax.set_xlabel(r'x [mm]', weight='bold')
ax.set_ylabel(r'y [mm]', rotation=0, weight='bold', position=(0,0.52))
stream_cbar_fluid = fig.colorbar(stream_fluid.lines, ax=ax)
stream_cbar_fluid.set_label(r'Speed fluid [mm/s]', weight='bold')
stream_cbar_gas = fig.colorbar(stream_gas.lines, ax=ax)
stream_cbar_gas.set_label(r'Speed gas [mm/s]', weight='bold')
ax.legend(bbox_to_anchor=(1.017,1.1))
fig.tight_layout()
plt.show()

