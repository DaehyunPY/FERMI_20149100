"""
Refer the details to the journal paper: PRL 115, 114801 (2015).
"""
import numpy as np
import xarray as xr
from scipy.special import jv

from .units import ALPHA, convert_units
from .electricfield import ElectricField

__all__ = ["EFTwinFermiPulses"]


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
                 t: np.ndarray,
                 nharmonic: int,
                 seed_k0: float,
                 seed_sigma: float,
                 seed_dt: float,
                 ds_strength: float,
                 ebeam_energy: float,
                 ebeam_sigma: float,
                 ebeam_chirp1: float,
                 ebeam_chirp2: float,
                 seed_phi: float = 0,
                 seed_ramp: float = 1,
                 ebeam_timing: float = 0,
                 ebeam_ramp: float = 1):
        """Initialize EFTwinFermiPulses.

        Args:
            nharmonic (int): Harmonic number.
            seed_k0 (float): Photon energy of the seeds.
            seed_sigma (float): Transform-limited pulse duration
                (standard deviation) of the seed intensities.
            seed_dt (float): Separation between the two seeds in time
                domain.
            seed_phi (float): Phase `phi` between the two seeds, which
                is defined by the following electric field:
                    f(t)**0.5 * sin(k * t)
                    + g(t - dt)**0.5 * sin(k * (t - dt) + phi),
                where `f` and `g` are the intensity enevelopes of the
                each seed and `dt` is the positive-valued delay.
            seed_ramp (float): Amplitude of the 2nd seed, relative to
                the 1st.
            ds_strength (float): Strength of the dispersive section.
            ebeam_timing (float): Relative timing of the electron beam
                to the 1st seed.
            ebeam_ramp (float): Amplitude of the energy modulation of
                the electron beam induced by the seeds.
        """
        self.__nharmonic = nharmonic
        self.__seed_k0 = seed_k0
        self.__seed_sigma = seed_sigma
        self.__seed_sigmaprime = seed_sigma * 2**0.5
        self.__seed_dt = seed_dt
        self.__seed_phi = seed_phi
        self.__seed_ramp = seed_ramp
        self.__ds_strength = ds_strength
        self.__ebeam_energy = ebeam_energy
        self.__ebeam_sigma = ebeam_sigma
        self.__ebeam_chirp1 = ebeam_chirp1
        self.__ebeam_chirp2 = ebeam_chirp2
        self.__ebeam_timing = ebeam_timing
        self.__ebeam_ramp = ebeam_ramp

        n = len(t)
        dt = t[1] - t[0]
        k0 = nharmonic * seed_k0
        k = k0 + np.fft.fftshift(np.fft.fftfreq(n, dt / 2 / np.pi))
        y = self.__bunching(t)
        z = (np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)))
             * dt / (2 * np.pi)**0.5)
        self.__samples = xr.Dataset({"at_t": (["t"], y), "at_k": (["k"], z)},
                                    coords={"t": (["t"], t), "k": (["k"], k)})

    @staticmethod
    def in_units(
            t: Tuple[np.ndarray, str],
            nharmonic: int,
            seed_k0: Tuple[float, str],
            seed_fwhm: Tuple[float, str],
            seed_dt: Tuple[float, str],
            ds_strength: Tuple[float, str],
            ebeam_energy: Tuple[float, str],
            ebeam_sigma: Tuple[float, str],
            ebeam_chirp1: Tuple[float, str],
            ebeam_chirp2: Tuple[float, str],
            seed_phi: Tuple[float, str] = (0, "rad"),
            seed_ramp: Tuple[float, str] = (1, "au"),
            ebeam_timing: Tuple[float, str] = (0, "au"),
            ebeam_ramp: Tuple[float, str] = (1, "au"),
    ) -> "EFTwinFermiPulses":
        """Initialize ElectricField with familiar units."""
        return EFTwinFermiPulses(
            t=convert_units(*t),
            nharmonic=nharmonic,
            seed_k0=convert_units(*seed_k0),
            seed_sigma=convert_units(*seed_fwhm) / (8 * np.log(2))**0.5,
            seed_dt=convert_units(*seed_dt),
            ds_strength=convert_units(*ds_strength),
            ebeam_energy=convert_units(*ebeam_energy),
            ebeam_sigma=convert_units(*ebeam_sigma),
            ebeam_chirp1=convert_units(*ebeam_chirp1),
            ebeam_chirp2=convert_units(*ebeam_chirp2),
            seed_phi=convert_units(*seed_phi),
            seed_ramp=convert_units(*seed_ramp),
            ebeam_timing=convert_units(*ebeam_timing),
            ebeam_ramp=convert_units(*ebeam_ramp),
        )

    def __seed_field(self, t: (float, np.ndarray)) -> (complex, np.ndarray):
        """Complex-valued electric field of the seeds in time domain."""
        return (seed(t, sigma=self.__seed_sigmaprime)
                + seed(t, sigma=self.__seed_sigmaprime,
                       dt=self.__seed_dt,
                       phi=self.__seed_phi,
                       amp=self.__seed_ramp))

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
        c = 1 / ALPHA  # speed of light in atomic units
        n = self.__nharmonic
        k0 = self.__seed_k0
        e0 = self.__ebeam_energy
        esigma = self.__ebeam_sigma
        r = self.__ds_strength
        z = self.__seed_field(t)  # dims: [t]
        e = self.__ebeam_profile(t)  # dims: [t]
        acap = self.__ebeam_ramp * np.abs(z)  # dims: [t]
        bcap = r * k0 / c * esigma / e0
        phis = np.unwrap(np.angle(z))  # dims: [t]
        phie = r * k0 / c * e / e0  # dims: [t]
        return (np.exp(-(n * bcap)**2 / 2)
                * jv(n, -n * bcap * acap)
                * np.exp(1j * n * (phis + phie)))

    @property
    def samples(self) -> xr.Dataset:
        return self.__samples

    def at_t(self, t: (float, np.ndarray)) -> np.ndarray:
        return self.__samples["at_t"].interp(t=t).values

    def at_k(self, k: (float, np.ndarray)) -> np.ndarray:
        return self.__samples["at_k"].interp(k=k).values
