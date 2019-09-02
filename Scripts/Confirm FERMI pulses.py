#%%
from matplotlib import pyplot as plt
import numpy as np

import simul2

#%%
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

#%%
plt.figure()
plt.xlabel("Time (fs)")
plt.ylabel("Intensity")
x = simul2.Q_(field.samples["t"].values, "a_u_time").m_as("fs")
z = field.samples["at_t"].values
plt.plot(x, np.abs(z)**2)
plt.xlim(-400, 400)
plt.yticks([0])
plt.grid(True)
plt.tight_layout()
plt.show()
del x, z

#%%
plt.figure()
plt.xlabel("Energy (nm)")
plt.ylabel("Intensity")
x = simul2.Q_(field.samples["k"].values, "hartree").to("nm", "spectroscopy").m
z = field.samples["at_k"]
plt.plot(x, np.abs(z)**2)
plt.xlim(50.5, 50.8)
plt.ylim(0, None)
plt.yticks([0])
plt.grid(True, axis="x")

plt.twinx()
plt.ylabel("Phase (deg)")
plt.plot(x, np.angle(z) / np.pi * 180, "grey")
plt.ylim(-180, 180)
plt.grid(True)
plt.tight_layout()
plt.show()
