#%%
import itertools

from matplotlib import pyplot as plt
import numpy as np

import simul2

#%%
field = simul2.EFTwinGaussianPulses.in_units(fwhm="55 fs",
                                             k0="24.472 eV",
                                             dt="299.473 fs",
                                             phi="80 deg")
n = 86001
t = simul2.Q_(np.linspace(-2000, 2300, n), "fs").to_base_units().m  # to au
dt = t[1] - t[0]
k = np.fft.fftshift(np.fft.fftfreq(n, dt / 2 / np.pi))
y = field.at_t(t)
# z = np.fft.fftshift(np.fft.fft(y))
z = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)))
del field, n, dt

plt.figure(figsize=[6, 8])
plt.subplot(211)
plt.xlabel("Time (fs)")
plt.ylabel("Light intensity")
x = simul2.Q_(t, "a_u_time").m_as("fs")
plt.plot(x, y**2)
plt.ylim(0, None)
plt.yticks([0])
plt.grid(True)
del x

plt.subplot(212)
plt.xlabel("Energy (eV)")
plt.ylabel("Phase (deg)")
x = simul2.Q_(k, "hartree").m_as("eV")
plt.plot(x, simul2.Q_(np.angle(z), "rad").m_as("deg"), "grey")
plt.xlim(24.3, 24.6)
plt.ylim(-180, 180)
plt.grid(True)

plt.twinx()
plt.ylabel("Light intensity")
plt.plot(x, np.abs(z)**2)
plt.grid(True)
plt.ylim(0, None)
plt.yticks([0])
plt.tight_layout()
plt.show()
del x

field = simul2.EFInterpolated(samples_at_t=y, samples_at_k=z, t=t, k=k)
del y, z, t, k

#%%
target = simul2.predefined_target("helium")
k = simul2.Q_(np.linspace(24.3, 24.6, 301), "eV").to_base_units().m  # to au
t = simul2.Q_(np.linspace(-100, 1000, 1101), "fs").to_base_units().m  # to au
wp = simul2.WavePacket(field, target)
y = wp(t)
n = 4
where = wp.status["freq"][wp.status["coeff"].abs().nlargest(n).index]

plt.figure(figsize=[6, 12])
plt.subplot(411)
plt.xlabel("Energy (eV)")
plt.ylabel("Strength n^(−3)")
x = simul2.Q_(target["level"].values, "hartree").m_as("eV")
plt.vlines(x, 0, target["strength"], "grey")
plt.xlim(24.3, 24.6)
plt.ylim(0, 0.003)
plt.yticks([0])
plt.grid(True)
del x

plt.twinx()
plt.ylabel("Light intensity")
x = simul2.Q_(k, "hartree").m_as("eV")
plt.plot(x, np.abs(field.at_k(k))**2)
plt.ylim(0, None)
plt.yticks([0])

plt.subplot(412)
plt.xlabel("Status")
plt.ylabel("Population")
sorted = where.sort_values().index
plt.bar(sorted, wp.status.loc[sorted, "coeff"].abs()**2)
plt.yticks([0])
plt.grid(True)
del x

plt.subplot(413)
plt.xlabel("Time (fs)")
plt.ylabel("Population")
x = simul2.Q_(t, "a_u_time").m_as("fs")
for i, j in itertools.combinations(where.sort_values().index, 2):
    plt.plot(x, y.loc[{"n": i, "n'": j}], label="{}, {}".format(i, j))
plt.yticks([0])
plt.grid(True)
plt.legend(loc="lower left", shadow=True)

plt.twinx()
plt.ylabel("Light intensity")
plt.plot(x, field.at_t(t)**2, "grey", alpha=0.5)
plt.ylim(0, None)
plt.yticks([0])

plt.subplot(414)
plt.xlabel("Time (fs)")
plt.ylabel("Total population")
plt.plot(x, y.sum(["n", "n'"]))
plt.ylim(0, None)
plt.yticks([0])
plt.grid(True)

plt.twinx()
plt.ylabel("Light intensity")
plt.plot(x, field.at_t(t)**2, "grey", alpha=0.5)
plt.ylim(0, None)
plt.yticks([0])
plt.tight_layout()
plt.show()
