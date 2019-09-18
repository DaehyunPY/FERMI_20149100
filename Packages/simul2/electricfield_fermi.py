"""
Refer the details to the journal paper: PRL 115, 114801 (2015).
"""
from typing import Any, Optional
import numpy as np
import xarray as xr
from scipy.special import jv

from .units import Q_, quantity
from .electricfield import ElectricField

__all__ = ["EFTwinFermiPulses"]

C = Q_("c").m_as("bohr / a_u_time")  # speed of light


def gauss(x: (float, np.ndarray),
          c: float = 1,
          b: float = 0,
          a: float = 1) -> (float, np.ndarray):
    """Gaussian function.

    Defined as:
        gauss(x) = exp(-x**2 / 2),
        gauss(x, c, b, a) = a * gauss((x - b) / c).
    Refer the detailed definition to the Wikipedia page below:
        https://en.wikipedia.org/wiki/Gaussian_function.
    """
    scaled = (x - b) / c
    return a * np.exp(-scaled**2 / 2)


def seed(t: (float, np.ndarray),
         sigma: float,
         k: float = 0,
         phi: float = 0,
         dt: float = 0,
         amp: float = 1) -> (complex, np.ndarray):
    delayed = t - dt
    return amp * gauss(delayed / sigma) * np.exp(1j * (k * delayed + phi))


class EFTwinFermiPulses(ElectricField):
    def __init__(self,
                 tsamples: np.ndarray,
                 nharmonic: int,
                 seed_k0: float,
                 seed_dt: float,
                 seed1st_sigma: float,
                 ds_strength: float,
                 ebeam_energy: float,
                 ebeam_sigma: float,
                 ebeam_chirp1: float,
                 ebeam_chirp2: float,
                 seed_phi: float = 0,
                 seed1st_ramp: float = 1,
                 seed2nd_sigma: Optional[float] = None,
                 seed2nd_ramp: float = 1,
                 ebeam_timing: float = 0,
                 ebeam_ramp: float = 1):
        """Initialize EFTwinFermiPulses.

        Args:
            nharmonic (int): Harmonic number.
            seed_k0 (float): Photon energy of the seeds.
            seed_dt (float): Separation between the two seeds in time
                domain.
            seed_phi (float): Phase `phi` between the two seeds, which
                is defined by the following electric field:
                    f(t)**0.5 * sin(k * t)
                    + g(t - dt)**0.5 * sin(k * (t - dt) + phi),
                where `f` and `g` are the intensity enevelopes of the
                each seed and `dt` is the positive-valued delay.
            seed1st_sigma (float): Transform-limited pulse duration
                (standard deviation) of the 1st seed intensities.
            seed1st_ramp (float): Relative amplitude of the 1st seed.
            seed2nd_sigma (float): Transform-limited pulse duration
                (standard deviation) of the 2nd seed intensities.
            seed2nd_ramp (float): Relative amplitude of the 2nd seed.
            ds_strength (float): Strength of the dispersive section.
            ebeam_energy (float): Electron beam nominal energy.
            ebeam_sigma (float): Electron beam energy spread.
            ebeam_chirp1 (float): Electron beam cubic chirp.
            ebeam_chirp2 (float): Electron beam quadratic chirp.
            ebeam_timing (float): Relative timing of the electron beam
                to the 1st seed.
            ebeam_ramp (float): Relative amplitude of the energy
                modulation of the electron beam induced by the seeds.

            All arguments accept only values in Hartree atomic units.
            Either are output values. To initialize an instance with
            your preferred units, use method EFTwinFermiPulses.in_units.

            Each argument is related to the following arguments in the
            MATLAB code from Primož Rebernik Ribič, FERMI:
                nharmonic = n
                seed_k0 = lambdaseed = lambdaFEL / n
                seed_dt = deltat
                seed_phi = deltaphi
                seed1st_sigma = tau10 / (8 * log(2))**0.5,
                    or seed1st_fwhm = tau10
                seed1st_ramp = C1
                seed2nd_sigma = tau20 / (8 * log(2))**0.5,
                    or seed2nd_fwhm = tau20
                seed2nd_ramp = C2
                ds_strength = R56
                ebeam_energy = E0
                ebeam_sigma = sigmaE
                ebeam_chirp1 = ebeamlinchirp
                ebeam_chirp2 = ebeamquadchirp
                ebeam_timing = ebeamtiming
                ebeam_ramp = A0
        """
        self.__nharmonic = nharmonic
        self.__seed_k0 = seed_k0
        self.__seed_dt = seed_dt
        self.__seed_phi = seed_phi
        self.__seed1st_sigma = seed1st_sigma
        self.__seed1st_sigmaprime = seed1st_sigma * 2**0.5
        self.__seed1st_ramp = seed1st_ramp
        if seed2nd_sigma is None:
            seed2nd_sigma = seed1st_sigma
        self.__seed2nd_sigma = seed2nd_sigma
        self.__seed2nd_sigmaprime = seed2nd_sigma * 2**0.5  
        self.__seed2nd_ramp = seed2nd_ramp
        self.__ds_strength = ds_strength
        self.__ebeam_energy = ebeam_energy
        self.__ebeam_sigma = ebeam_sigma
        self.__ebeam_chirp1 = ebeam_chirp1
        self.__ebeam_chirp2 = ebeam_chirp2
        self.__ebeam_timing = ebeam_timing
        self.__ebeam_ramp = ebeam_ramp

        n = len(tsamples)
        dt = tsamples[1] - tsamples[0]
        k0 = nharmonic * seed_k0
        ksamples = k0 + np.fft.fftshift(np.fft.fftfreq(n, dt / 2 / np.pi))
        y = self.__bunching(tsamples)
        z = (np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)))
             * dt / (2 * np.pi)**0.5)
        self.__samples = xr.Dataset({"at_t": (["t"], y), "at_k": (["k"], z)},
                                    coords={"t": (["t"], tsamples),
                                            "k": (["k"], ksamples)})

    @staticmethod
    def in_units(
            tsamples: Any,
            nharmonic: Any,
            seed_k0: Any,
            seed_dt: Any,
            seed1st_fwhm: Any,
            ds_strength: Any,
            ebeam_energy: Any,
            ebeam_sigma: Any,
            ebeam_chirp1: Any,
            ebeam_chirp2: Any,
            seed_phi: Any = 0,
            seed1st_ramp: Any = 1,
            seed2nd_fwhm: Any = None,
            seed2nd_ramp: float = 1,
            ebeam_timing: Any = 0,
            ebeam_ramp: Any = 1,
    ) -> "EFTwinFermiPulses":
        """Initialize ElectricField with familiar units."""
        if seed2nd_fwhm is None:
            seed2nd_fwhm = seed1st_fwhm
        return EFTwinFermiPulses(
            tsamples=quantity(tsamples, fr_default="a_u_time"),
            nharmonic=quantity(nharmonic, fr_default=""),
            seed_k0=quantity(seed_k0, fr_default="hartree",
                             context="spectroscopy"),
            seed_dt=quantity(seed_dt, fr_default="a_u_time"),
            seed_phi=quantity(seed_phi, fr_default="rad"),
            seed1st_sigma=quantity(seed1st_fwhm, fr_default="a_u_time")
                          / (8 * np.log(2))**0.5,
            seed1st_ramp=quantity(seed1st_ramp, fr_default=""),
            seed2nd_sigma=quantity(seed2nd_fwhm, fr_default="a_u_time")
                          / (8 * np.log(2))**0.5,
            seed2nd_ramp=quantity(seed2nd_ramp, fr_default=""),
            ds_strength=quantity(ds_strength, fr_default="bohr"),
            ebeam_energy=quantity(ebeam_energy, fr_default="hartree"),
            ebeam_sigma=quantity(ebeam_sigma, fr_default="hartree"),
            ebeam_chirp1=quantity(ebeam_chirp1,
                                  fr_default="hartree / a_u_time"),
            ebeam_chirp2=quantity(ebeam_chirp2,
                                  fr_default="hartree / a_u_time**2"),
            ebeam_timing=quantity(ebeam_timing, fr_default="a_u_time"),
            ebeam_ramp=quantity(ebeam_ramp, fr_default=""),
        )

    def __seed_field(self, t: (float, np.ndarray)) -> (complex, np.ndarray):
        """Complex-valued electric field of the seeds in time domain."""
        return (
            seed(t, sigma=self.__seed1st_sigmaprime, amp=self.__seed1st_ramp)
            + seed(t, sigma=self.__seed2nd_sigmaprime, amp=self.__seed2nd_ramp,
                   dt=self.__seed_dt, phi=self.__seed_phi)
        )

    def __ebeam_profile(self, t: (float, np.ndarray)) -> (float, np.ndarray):
        """Time-dependent energy profile imprinted onto the electron
        beam by the linear acceleator."""
        delayed = t - self.__ebeam_timing
        c0 = self.__ebeam_energy
        c1 = self.__ebeam_chirp1
        c2 = self.__ebeam_chirp2
        return c0 + c1 * delayed + 0.5 * c2 * delayed**2

    def __bunching(self, t: (float, np.ndarray)) -> (complex, np.ndarray):
        """Bunching factor."""
        n = self.__nharmonic
        k0 = self.__seed_k0
        e0 = self.__ebeam_energy
        esigma = self.__ebeam_sigma
        r = self.__ds_strength
        z = self.__seed_field(t)  # dims: [t]
        e = self.__ebeam_profile(t)  # dims: [t]
        acap = self.__ebeam_ramp * np.abs(z)  # dims: [t]
        bcap = r * k0 / C * esigma / e0
        phis = np.unwrap(np.angle(z))  # dims: [t]
        phie = r * k0 / C * e / e0  # dims: [t]
        return (np.exp(-(n * bcap)**2 / 2)
                * jv(n, -n * bcap * acap)
                * np.exp(1j * n * (phis + phie)))

    @property
    def samples(self) -> xr.Dataset:
        return self.__samples

    def at_t(self, t: (float, np.ndarray)) -> (complex, np.ndarray):
        return self.__samples["at_t"].interp(t=t).values

    # def sqr_at_t(self, t: (float, np.ndarray)) -> (float, np.ndarray):
    #     z = self.__samples["at_t"].interp(t=t).values
    #     return np.abs(z)**2

    def at_k(self, k: (float, np.ndarray)) -> (complex, np.ndarray):
        return self.__samples["at_k"].interp(k=k).values

    # def sqr_at_k(self, k: (float, np.ndarray)) -> (float, np.ndarray):
    #     z = self.__samples["at_k"].interp(k=k).values
    #     return np.abs(z)**2
