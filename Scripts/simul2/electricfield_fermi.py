"""
Refer the details to the journal paper: PRL 115, 114801 (2015).
"""
from typing import Any
import numpy as np
import xarray as xr
from scipy.special import jv

from .units import Q_
from .electricfield import ElectricField

__all__ = ["EFTwinFermiPulses"]

C = Q_("c").to_base_units().m  # speed of light


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
            seed_fwhm: Any,
            seed_dt: Any,
            ds_strength: Any,
            ebeam_energy: Any,
            ebeam_sigma: Any,
            ebeam_chirp1: Any,
            ebeam_chirp2: Any,
            seed_phi: Any = 0,
            seed_ramp: Any = 1,
            ebeam_timing: Any = 0,
            ebeam_ramp: Any = 1,
    ) -> "EFTwinFermiPulses":
        """Initialize ElectricField with familiar units."""
        if (isinstance(tsamples, tuple)
                and len(tsamples) == 2
                and isinstance(tsamples[1], str)):
            t = Q_(*tsamples).to_base_units()
        else:
            t = Q_(tsamples).to_base_units()
        nharmonic = Q_(nharmonic).to_base_units()
        seed_k0 = Q_(seed_k0)
        if not seed_k0.unitless:
            seed_k0.ito("hartree", "spectroscopy")
        seed_k0.ito_base_units()
        seed_fwhm = Q_(seed_fwhm).to_base_units()
        seed_dt = Q_(seed_dt).to_base_units()
        ds_strength = Q_(ds_strength).to_base_units()
        ebeam_energy = Q_(ebeam_energy).to_base_units()
        ebeam_sigma = Q_(ebeam_sigma).to_base_units()
        ebeam_chirp1 = Q_(ebeam_chirp1).to_base_units()
        ebeam_chirp2 = Q_(ebeam_chirp2).to_base_units()
        seed_phi = Q_(seed_phi).to_base_units()
        seed_ramp = Q_(seed_ramp).to_base_units()
        ebeam_timing = Q_(ebeam_timing).to_base_units()
        ebeam_ramp = Q_(ebeam_ramp).to_base_units()
        if not ((t.check("[time]") or t.unitless)
                and nharmonic.dimensionless
                and (seed_fwhm.check("[time]") or seed_fwhm.unitless)
                and (seed_dt.check("[time]") or seed_dt.unitless)
                and (ds_strength.check("[length]") or ds_strength.unitless)
                and (ebeam_energy.check("[energy]") or ebeam_energy.unitless)
                and (ebeam_sigma.check("[energy]") or ebeam_sigma.unitless)
                and (ebeam_chirp1.check("[energy] / [time]")
                     or ebeam_chirp1.unitless)
                and (ebeam_chirp2.check("[energy] / [time]**2")
                     or ebeam_chirp2.unitless)
                and seed_phi.dimensionless
                and seed_ramp.dimensionless
                and (ebeam_timing.check("[time]") or ebeam_timing.unitless)
                and ebeam_ramp.unitless):
            raise ValueError("An assigned dimension is mismatched.")
        return EFTwinFermiPulses(
            tsamples=t.m,
            nharmonic=nharmonic.m,
            seed_k0=seed_k0.m,
            seed_sigma=seed_fwhm.m / (8 * np.log(2))**0.5,
            seed_dt=seed_dt.m,
            ds_strength=ds_strength.m,
            ebeam_energy=ebeam_energy.m,
            ebeam_sigma=ebeam_sigma.m,
            ebeam_chirp1=ebeam_chirp1.m,
            ebeam_chirp2=ebeam_chirp2.m,
            seed_phi=seed_phi.m,
            seed_ramp=seed_ramp.m,
            ebeam_timing=ebeam_timing.m,
            ebeam_ramp=ebeam_ramp.m,
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

    def at_t(self, t: (float, np.ndarray)) -> np.ndarray:
        return self.__samples["at_t"].interp(t=t).values

    def at_k(self, k: (float, np.ndarray)) -> np.ndarray:
        return self.__samples["at_k"].interp(k=k).values
