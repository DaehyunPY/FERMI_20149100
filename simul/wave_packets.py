from functools import lru_cache

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
    def __init__(self, sigma: float, k0: float, dt: float, phi: float, amp: float = 1):
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
    def in_experimental_units(sigma: float, k0: float, dt: float, phi: float, amp: float = 1) -> 'HeWavePackets':
        """
        :param sigma: fwhm of a pulse in femto-sec
        :param k0: eV
        :param dt: femto-sec
        :param phi: deg
        :param amp: 1
        :return: WavePackets
        """
        return HeWavePackets(
            sigma = in_femto_sec(sigma) / (8 * log(2)) ** 0.5,
            k0 = in_electron_volt(k0),
            dt = in_femto_sec(dt),
            phi = in_degree(phi),
            amp = amp,
        )

    @property
    def sigma(self) -> float:
        return self.__sigma

    @property
    def k0(self) -> float:
        return self.__k0

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
            self.amp * sin(self.k0*t) * gauss(t, self.sigma)
            + self.amp * sin(self.k0*(t-self.dt)-self.phi) * gauss((t-self.dt), self.sigma)
        )

    def pulsesabc(self, t: (float, ndarray)) -> (float, ndarray):
        return (
            self.amp * gauss(t, self.sigma)
            + self.amp * gauss((t-self.dt), self.sigma)
        )

    def pulses_k(self, k: (float, ndarray)) -> (float, ndarray):
        return (
            # 1st term
            self.amp / 2j / self.sigma
            * (1 + exp(-1j * (k * self.dt + self.phi)))
            * gauss(k - self.k0, 1 / self.sigma)
            # # 2nd term (you can ignore it)
            # - self.amp / 2j / self.sigma
            # * (1 + exp(-1j * (k * self.dt - self.phi)))
            # * gauss(k + self.k0, 1 / self.sigma)
        )

    @lru_cache(maxsize=None)
    def target_poplev(self) -> ndarray:
        return self.target_nlev ** -1.5 * self.pulses_k(self.target_klev)  # shape: n

    def __call__(self, t: ndarray) -> ndarray:
        pop = self.target_poplev()  # shape: n
        wave = -1j * exp(1j * self.target_klev[None, :] * t[:, None]) * pop[None, :]  # shape: (t,n)
        return (wave[:, :, None] * wave[:, None, :].conj()).real  # shape: (t,n,n)
