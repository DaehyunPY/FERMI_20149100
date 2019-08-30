#%%
from matplotlib import pyplot as plt
import numpy as np

import simul2

#%% Confirm simul2.sqr_at_t method
field = simul2.EFTwinGaussianPulses.in_units(fwhm=100, k0=0.1, dt=400)
n = 601
t = np.linspace(-100, 500, n)
y = field.at_t(t)

plt.figure()
plt.xlabel("Time")
plt.ylabel("Light intensity")
plt.plot(t, y**2)
plt.plot(t, field.sqr_at_t(t))
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Confirm simul2.at_k and simul2.sqr_at_k methods
field = simul2.EFTwinGaussianPulses.in_units(fwhm=1, k0=10, dt=5)
n = 4501
t = np.linspace(-20, 25, n)
dt = t[1] - t[0]
k = np.fft.fftshift(np.fft.fftfreq(n, dt / 2 / np.pi))
y = field.at_t(t)
# z = np.fft.fftshift(np.fft.fft(y)) * dt / (2 * np.pi)**0.5
z = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y))) * dt / (2 * np.pi)**0.5

plt.figure(figsize=[6, 8])
plt.subplot(311)
plt.xlabel("Time")
plt.ylabel("Electric field")
plt.plot(t, y)
plt.grid(True)
plt.xlim(-5, 10)
plt.yticks([0])

plt.subplot(312)
plt.xlabel("Energy")
plt.ylabel("Phase (deg)")
converted = simul2.Q_(np.angle(field.at_k(k)), "rad").to("deg")
plt.plot(k, converted, "grey")
plt.xlim(-20, 20)
plt.ylim(-180, 180)
plt.grid(True)

plt.twinx()
plt.ylabel("Light intensity")
plt.plot(k, np.abs(field.at_k(k))**2)
plt.plot(k, field.sqr_at_k(k))
plt.ylim(0, None)
plt.yticks([0])

plt.subplot(313)
plt.xlabel("Energy")
plt.ylabel("Phase (deg)")
converted = simul2.Q_(np.angle(z), "rad").to("deg")
plt.plot(k, converted, "grey")
plt.xlim(-20, 20)
plt.ylim(-180, 180)
plt.grid(True)

plt.twinx()
plt.ylabel("Light intensity")
plt.plot(k, np.abs(z)**2)
plt.ylim(0, None)
plt.yticks([0])
plt.tight_layout()
plt.show()
