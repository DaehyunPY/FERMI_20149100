"""
PRL 115, 114801 (2015)

Please keep the Python style guide of PEP8: pep8.org.
"""
# %%
import numpy as np
from scipy.special import jv

# %%
# Constants
C = 299792458
EV = 1.60217662e-19

# Machine parameters, to be checked from logbook
C1 = 1
C2 = 0.87
lambdaFEL = 50.52e-9 + 0.07e-9

# Other parameters
E0 = 1.16867e9 * EV  # electron beam nominal energy (J)
sigmaE = 150e3 * EV  # electron beam energy spread (J)
R56 = 50e-6  # dispersive strength
ebeamlinchirp = 0.19e6 * EV / 1e-12  # electron beam cubic chirp
ebeamquadchirp = 5.42e6 * EV / 1e-12 ** 2  # electron beam quadratic chirp

n = 5  # harmonic number
lambdaseed = lambdaFEL * n  # seed laser wavelength
k1 = 2 * np.pi / lambdaseed  # seed laser wave number

tau10 = 130e-15  # first seed transform-limited pulse duration
GDD1 = 0  # first seed linear frequency (quadratic phase) chirp
tau1 = (1 + (4*np.log(2)*GDD1/tau10**2) ** 2) ** 0.5 * tau10

tau20 = tau10  # second seed transform-limited pulse duration
GDD2 = 0  # second seed linear frequency (quadratic phase) chirp
tau2 = (1 + (4*np.log(2)*GDD2/tau20**2) ** 2) ** 0.5 * tau20

deltat = 150e-15  # separation between the seeds


def output(t: (float, np.ndarray)) -> (float, np.ndarray):
    Psi1 = 1 / (2*GDD1 + tau10**4/(8*np.log(2)**2*GDD1)) * t ** 2
    Psi2 = 1 / (2*GDD2 + tau20**4/(8*np.log(2)**2*GDD2)) * (t - deltat) ** 2

    deltaphi = 3.146894088480846
    ebeamtiming = 1.966066329749903e-12
    seedfield = (
        C1 * np.exp(-2*np.log(2)*t**2/tau1**2) * np.exp(1j*Psi1)
        + C2 * np.exp(-2*np.log(2)*(t-deltat)**2/tau2**2) * np.exp(1j*Psi2) * np.exp(1j*deltaphi))  # seed electric field; first seed sentered at time=0 fs
    seedenvelope = np.abs(seedfield) ** 2  # seed envelope
    seedphase = np.unwrap(np.angle(seedfield))  # seed phase

    A0 = 3  # amplitude of the energy modulation of the electron beam induced by the seeds
    A = A0 * seedenvelope ** 0.5
    B = R56 * k1 * sigmaE / E0  # normalized dispersive strength
    ebeamenergyprofile = (
        E0
        + ebeamlinchirp * (t - ebeamtiming)
        + (1/2) * ebeamquadchirp * (t - ebeamtiming) ** 2
    )
    # electorn beam energy profile induces a phase onto the FEL pulse
    ebeamphase = B / sigmaE * ebeamenergyprofile

    # bunching (proportional to the FEL electric field) in the time domain
    return (np.exp(-(n*B)**2/2)
            * jv(n, -n*B*A)
            * np.exp(1j*n*seedphase)
            * np.exp(1j*n*ebeamphase))


# %%
t = np.linspace(-5.125e-12, 5.275e-12, 2 ** 12, endpoint=False)
wave = output(t)
freq = C * n / lambdaseed + np.fft.fftshift(np.fft.fftfreq(t.shape[0], t[1] - t[0]))
x = C / freq * 1e9
y = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(wave)))) ** 2

# %%
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.xlim(50.5, 50.8)
plt.grid(True)
plt.show()
