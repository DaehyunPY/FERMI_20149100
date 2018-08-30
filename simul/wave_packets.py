from functools import lru_cache
from typing import Optional

from pandas import read_excel
from numpy import sin, pi, exp, log, ndarray, gradient, concatenate
from importlib_resources import path
from numba import jit

from . import rsc
from .units import in_femto_sec, in_electron_volt, in_degree


__all__ = [
    'gauss', 'ispeak', 'HeWavePackets',
]


with path(rsc, 'he_levels.xlsx') as fp:
    target = read_excel(fp, 'He levels')


@jit
def gauss(x: float, sigma: float) -> float:
    return 1 / (2*pi)**0.5 * exp(-(x/sigma)**2/2)


@jit
def ispeak(arr: ndarray) -> ndarray:
    diff = gradient(arr)  # shape: k
    ispos = 0 < diff  # shape: k
    return concatenate([[False], ispos[:-1] & ~ispos[1:]])  # shape: k


class HeWavePackets:
    def __init__(self,
                 sigma: float,
                 k0: float,
                 dt: float = 0,
                 phi: float = 0,
                 amp: float = 1):
        """
        Omega-omega experiment with atomic helium. All the parameters are in atomic units!
        Omega-omega pulses := amp * sin(k0*t) * gauss(t, sigma) + amp * np.sin(k0*(t-dt)-phi) * gauss((t-dt), sigma)
        :param sigma:
        :param k0:
        :param dt:
        :param phi:
        :param amp:
        """
        self.__sigma = sigma
        self.__k0 = k0
        self.__dt = dt
        self.__phi = phi
        self.__amp = amp

    @staticmethod
    def in_experimental_units(fwhm: float,
                              k0: float,
                              dt: float = 0,
                              phi: float = 0,
                              amp: float = 1) -> 'HeWavePackets':
        """
        :param fwhm: fwhm of pulse intensity in femto-sec
        :param k0: in eV
        :param dt: in femto-sec
        :param phi: in degree
        :return: WavePackets
        """
        return HeWavePackets(
            sigma = in_femto_sec(fwhm) / (8 * log(2))**0.5 * 2**0.5,
            k0 = in_electron_volt(k0),
            dt = in_femto_sec(dt),
            phi = in_degree(phi),
            amp = amp,
        )

    @property
    def sigma(self) -> float:
        return self.__sigma

    @property
    def tdim_sigma(self) -> float:
        return self.__sigma

    @property
    def tdim_fwhm(self) -> float:
        return self.__sigma * (8 * log(2))**0.5 / 2**0.5

    @property
    def kdim_sigma(self) -> float:
        return 1 / self.__sigma

    @property
    def kdim_fwhm(self) -> float:
        return 1 / self.__sigma * (8 * log(2))**0.5 / 2**0.5

    @property
    def k0(self) -> float:
        return self.__k0

    @property
    def dk(self) -> Optional[float]:
        if self.__dt == 0:
            return None
        return 2 * pi / self.__dt

    @property
    def dt(self) -> float:
        return self.__dt

    @property
    def phi(self) -> float:
        return self.__phi

    @property
    def amp(self) -> float:
        return self.__amp

    @property
    def target_nlev(self) -> ndarray:
        return target['n'].values

    @property
    def target_klev(self) -> ndarray:
        return target['level'].values

    def pulses(self, t: (float, ndarray)) -> (float, ndarray):
        return (
            self.amp * sin(self.k0 * t) * gauss(t, self.tdim_sigma)
            + self.amp * sin(self.k0 * (t - self.dt) - self.phi) * gauss((t - self.dt), self.tdim_sigma)
        )

    def pulsesabc(self, t: (float, ndarray)) -> (float, ndarray):
        return (
            self.amp * gauss(t, self.tdim_sigma)
            + self.amp * gauss((t-self.dt), self.tdim_sigma)
        )

    def pulses_k(self, k: (float, ndarray)) -> (float, ndarray):
        return (
            # 1st term
            self.amp * self.kdim_sigma / 2j
            * (1 + exp(-1j * (k * self.dt + self.phi)))
            * gauss(k - self.k0, self.kdim_sigma)
            # # 2nd term (you can ignore it)
            # - self.amp * self.kdim_sigma / 2j
            # * (1 + exp(-1j * (k * self.dt - self.phi)))
            # * gauss(k + self.k0, self.kdim_sigma)
        )

    @lru_cache(maxsize=None)
    def target_poplev(self) -> ndarray:
        return self.target_nlev ** -1.5 * self.pulses_k(self.target_klev)  # shape: n

    def __call__(self, t: ndarray) -> ndarray:
        pop = self.target_poplev()  # shape: n
        wave = -1j * exp(1j * self.target_klev[None, :] * t[:, None]) * pop[None, :]  # shape: (t,n)
        return (wave[:, :, None] * wave[:, None, :].conj()).real  # shape: (t,n,n)
