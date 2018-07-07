from numpy import linspace, pi
import matplotlib.pyplot as plt


# %%
dff = linspace(0, 0.1, 1000)/27.2116
tau = 2*pi/dff

plt.figure(figsize=(8, 8))
plt.plot(tau*0.02418884326505, dff*27.2116)
plt.minorticks_on()
plt.grid(which='both')
plt.xlim(0, 1000)
plt.ylim(0, 0.04)
plt.xlabel("Kepler orbit time (fs)")
plt.ylabel("differential energy (eV)")
plt.tight_layout()
plt.savefig("map_time_to_energy.pdf")
plt.show()
