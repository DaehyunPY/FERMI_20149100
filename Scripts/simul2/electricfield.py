from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import xarray as xr

from .units import convert_units

__all__ = [
    "ElectricField",
    "EFInterpolated",
    "EFGaussianPulse",
    "EFTwinGaussianPulses",
]


class ElectricField(ABC):
    @abstractmethod
    def at_t(self, t: (float, np.ndarray)) -> (float, np.ndarray):
        pass

    @abstractmethod
    def at_k(self, k: (complex, np.ndarray)) -> (complex, np.ndarray):
        pass


class EFInterpolated(ElectricField):
    def __init__(self,
                 samples_at_t: np.ndarray,
                 samples_at_k: np.ndarray,
                 t: np.ndarray,
                 k: np.ndarray):
        self.__samples = xr.Dataset(
            {"at_t": (["t"], samples_at_t),
             "at_k": (["k"], samples_at_k)},
            coords={"t": (["t"], t),
                    "k": (["k"], k)},
        )

    @property
    def samples(self) -> xr.Dataset:
        return self.__samples

    def at_t(self, t: (float, np.ndarray)) -> np.ndarray:
        return self.__samples["at_t"].interp(t=t).values

    def at_k(self, k: (float, np.ndarray)) -> np.ndarray:
        return self.__samples["at_k"].interp(k=k).values


def gauss(x: (float, np.ndarray),
          sigma: float = 1,
          mu: float = 0,
          amp: float = 1) -> (float, np.ndarray):
    """Gaussian function.

    Defined as:
        gauss(x, sigma) = 1 / (2 * pi)**0.5 / sigma
                          * exp(-(x / sigma)**2 / 2),
        gauss(x, sigma, mu, amp) = amp * gauss(x - mu, sigma).
    Refer the detailed definition to the Wikipedia page below:
        https://en.wikipedia.org/wiki/Gaussian_function.
    """
    return amp / (2 * np.pi)**0.5 / sigma * np.exp(-((x - mu) / sigma)**2 / 2)


class EFGaussianPulse(ElectricField):
    """ElectricField of which envelope is shaped as a Gaussian function.

    Various functions in time and frequency domains:
        E(t) = amp' * sin(k0 * t) * gauss(t, sigma'),
        Envelope of E**2(t) = amp * gauss(t, sigma),
        E(k) ~ amp' / 2j / sigma' * gauss(k - k0, 1 / sigma'),
        abs(E)**2(k) ~ amp / 4 * gauss(k - k0, 1 / 2 / sigma),
    where
        sigma' = 2**0.5 * sigma,
        amp' = (8 * pi * amp**2 * sigma**2)**0.25.
    """
    def __init__(self, sigma: float, k0: float, amp: float = 1):
        """Initialize a Gaussian Pulse E(t)."""
        self.__sigma = sigma
        self.__sigmaprime = 2**0.5 * sigma
        self.__k0 = k0
        self.__amp = amp
        self.__ampprime = (8 * np.pi * amp**2 * sigma**2)**0.25

    @staticmethod
    def in_units(fwhm: Tuple[float, str],
                 k0: Tuple[float, str],
                 amp: Tuple[float, str] = (1, "au")) -> "EFGaussianPulse":
        """Initialize ElectricField with familiar units."""
        return EFGaussianPulse(
            sigma=convert_units(*fwhm) / (8 * np.log(2))**0.5,
            k0=convert_units(*k0),
            amp=convert_units(*amp),
        )

    def at_t(self, t: (float, np.ndarray)) -> (float, np.ndarray):
        """Electric field in time domain."""
        sigma = self.__sigmaprime
        k0 = self.__k0
        amp = self.__ampprime
        return amp * np.sin(k0 * t) * gauss(t, sigma)

    def sqr_at_t(self, t: (float, np.ndarray)) -> (float, np.ndarray):
        """Envelope of intensity in time domain."""
        sigma = self.__sigma
        amp = self.__amp
        return amp * gauss(t, sigma)

    def at_k(self, k: (complex, np.ndarray)) -> (complex, np.ndarray):
        """Electric field in frequency domain."""
        sigma = self.__sigmaprime
        k0 = self.__k0
        amp = self.__ampprime
        return (
            amp / 2j / sigma * gauss(k - k0, 1 / sigma)  # The 1st term
            # - amp / 2j / sigma * gauss(k + k0, 1 / sigma)  # The 2nd term
        )

    def sqr_at_k(self, k: (float, np.ndarray)) -> (float, np.ndarray):
        """Envelope of intensity in frequency domain."""
        sigma = self.__sigma
        k0 = self.__k0
        amp = self.__amp
        return amp / 4 * gauss(k - k0, 1 / 2 / sigma)


class EFTwinGaussianPulses(ElectricField):
    """ElectricField of which envelope is shaped as sum of two Gaussian
    functions in time domain.

    Various functions in time and frequency domains:
        E(t) = amp' * sin(k0 * t) * gauss(t, sigma')
               + amp' * sin(k0 * (t - dt) - phi)
               * gauss(t - dt, sigma'),
        Envelope of E**2(t) = amp * gauss(t, sigma)
                              + amp * gauss(t - dt, sigma),
        E(k) ~ amp' / 2j / sigma'
               * (1 + exp(-1j * (k * dt + phi)))
               * gauss(k - k0, 1 / sigma'),
        abs(E)**2(k) ~ amp * cos**2((k * dt + phi) / 2)
                       * gauss(k - k0, 1 / 2 / sigma),
    where
        sigma' = 2**0.5 * sigma,
        amp' = (8 * pi * amp**2 * sigma**2)**0.25.
    """
    def __init__(self,
                 sigma: float,
                 k0: float,
                 dt: float,
                 phi: float = 0,
                 amp: float = 1):
        """Initialize a Gaussian Pulse E(t)."""
        self.__sigma = sigma
        self.__sigmaprime = 2**0.5 * sigma
        self.__k0 = k0
        self.__dt = dt
        self.__phi = phi
        self.__amp = amp
        self.__ampprime = (8 * np.pi * amp**2 * sigma**2)**0.25

    @staticmethod
    def in_units(fwhm: Tuple[float, str],
                 k0: Tuple[float, str],
                 dt: Tuple[float, str],
                 phi: Tuple[float, str] = (0, "rad"),
                 amp: Tuple[float, str] = (1, "au")) -> "EFTwinGaussianPulses":
        """Initialize ElectricField with familiar units."""
        return EFTwinGaussianPulses(
            sigma=convert_units(*fwhm) / (8 * np.log(2))**0.5,
            k0=convert_units(*k0),
            dt=convert_units(*dt),
            phi=convert_units(*phi),
            amp=convert_units(*amp),
        )

    def at_t(self, t: (float, np.ndarray)) -> (float, np.ndarray):
        """Electric field in time domain."""
        sigma = self.__sigmaprime
        k0 = self.__k0
        dt = self.__dt
        phi = self.__phi
        amp = self.__ampprime
        return (
            amp * np.sin(k0 * t) * gauss(t, sigma)
            + amp * np.sin(k0 * (t - dt) - phi) * gauss(t - dt, sigma)
        )

    def sqr_at_t(self, t: (float, np.ndarray)) -> (float, np.ndarray):
        """Envelope of intensity in time domain."""
        sigma = self.__sigma
        dt = self.__dt
        amp = self.__amp
        return amp * gauss(t, sigma) + amp * gauss(t - dt, sigma)

    def at_k(self, k: (complex, np.ndarray)) -> (complex, np.ndarray):
        """Electric field in frequency domain."""
        sigma = self.__sigmaprime
        k0 = self.__k0
        dt = self.__dt
        phi = self.__phi
        amp = self.__ampprime
        return (
            amp / 2j / sigma  # The 1st term
            * (1 + np.exp(-1j * (k * dt + phi)))
            * gauss(k - k0, 1 / sigma)
            # - amp / 2j / sigma  # The 2nd term
            # * (1 + np.exp(-1j * (k * dt - phi)))
            # * gauss(k + k0, 1 / sigma)
        )

    def sqr_at_k(self, k: (float, np.ndarray)) -> (float, np.ndarray):
        """Envelope of intensity in frequency domain."""
        sigma = self.__sigma
        k0 = self.__k0
        dt = self.__dt
        phi = self.__phi
        amp = self.__amp
        return (
            amp * np.cos((k * dt + phi) / 2)**2
            * gauss(k - k0, 1 / 2 / sigma)
        )
