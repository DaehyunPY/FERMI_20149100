#%%
import itertools

from matplotlib import pyplot as plt
import numpy as np

import simul2

#%%
# For the detailed explaination of the arguments below,
# please read the docstring of simul2.EFTwinFermiPulses and the MATLAB code
# from Primož Rebernik Ribič, FERMI.
#
# A class simul2.Q_ is generated for unit conversion using package pint:
# https://pint.readthedocs.io/. The default unit system is Hartree atomic units.
#
field = simul2.EFTwinFermiPulses.in_units(
    tsamples=(np.linspace(-5.125, 5.275, 2**12, endpoint=False), "ps"),
    nharmonic=5,
    seed_k0="(50.52 + 0.07) * 5 nm",
    seed_dt="150 fs",
    seed_phi="3.146894088480846 rad",
    seed1st_fwhm="130 fs",
    seed1st_ramp=1,
    seed2nd_fwhm="130 fs",
    seed2nd_ramp=0.87,
    ds_strength="50 um",
    ebeam_energy="1.16867 GeV",
    ebeam_sigma="150 keV",
    ebeam_chirp1="0.19 MeV / ps",
    ebeam_chirp2="5.42 MeV / ps**2",
    ebeam_timing="1.966066329749903 ps",
    ebeam_ramp=3,
)
target = simul2.predefined_target("helium")
wp = simul2.WavePacket(field, target)
k = simul2.Q_(np.linspace(24.3, 24.6, 301), "eV").to_base_units().m
t = simul2.Q_(np.linspace(-100, 1000, 1101), "fs").to_base_units().m
y = wp(t)
n = 4  # Number of configurations to display
where = wp.status["freq"][wp.status["coeff"].abs().nlargest(n).index]

#%%
plt.figure(figsize=[6, 12])
plt.subplot(411)
plt.xlabel("Energy (eV)")
plt.ylabel("Strength n^(−3) / Light intensity")
x = simul2.Q_(target["level"].values, "hartree").m_as("eV")
plt.vlines(x, 0, target["strength"], "black")
plt.xlim(24.3, 24.6)
plt.ylim(0, 0.003)
plt.yticks([0])
plt.grid(True, axis="x")
del x

plt.twinx()
x = simul2.Q_(k, "hartree").m_as("eV")
plt.plot(x, np.abs(field.at_k(k))**2)
plt.ylim(0, None)
plt.yticks([0])

plt.twinx()
plt.ylabel("Phase (deg)")
x = simul2.Q_(k, "hartree").m_as("eV")
plt.plot(x, np.angle(field.at_k(k)) / np.pi * 180, "grey")
plt.ylim(-180, 180)
plt.grid(True)

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
plt.xlim(x[0], x[-1])
plt.yticks([0])
plt.grid(True)
plt.legend(loc="lower left", shadow=True)

plt.twinx()
plt.ylabel("Light intensity")
plt.plot(x, np.abs(field.at_t(t))**2, "grey")
plt.ylim(0, None)
plt.yticks([0])

plt.subplot(414)
plt.xlabel("Time (fs)")
plt.ylabel("Total population")
plt.plot(x, y.sum(["n", "n'"]))
plt.xlim(x[0], x[-1])
plt.ylim(0, None)
plt.yticks([0])
plt.grid(True)

plt.twinx()
plt.ylabel("Light intensity")
plt.plot(x, np.abs(field.at_t(t))**2, "grey")
plt.ylim(0, None)
plt.yticks([0])
plt.tight_layout()
plt.show()
