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
plt.plot(x, field.sqr_at_k(k))
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
plt.plot(x, field.sqr_at_t(t), "grey")
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
plt.plot(x, field.sqr_at_t(t), "grey")
plt.ylim(0, None)
plt.yticks([0])
plt.tight_layout()
plt.show()
