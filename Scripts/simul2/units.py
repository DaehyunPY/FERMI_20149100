from math import pi

from numba import jit

__all__ = ["convert_units"]


MA = 1.66053892173e-27  # kg, 1 dalton
C = 299792458  # m/s, speed of light
ME = 9.1093829140e-31  # kg, electron rest mass
E = 1.60217656535e-19  # C, elementary charge
HBAR = 1.05457172647e-34  # J*s, reduced Planck constant
KE = 8.9875517873681e9  # N*m2/C2, Coulomb constant
ALPHA = KE * E**2 / HBAR / C  # fine-structure constant
BOHR = HBAR / ME / C / ALPHA  # m, Bohr radius
HARTREE = ALPHA**2 * ME * C**2  # J, Hartree energy


@jit(nopython=True, nogil=True)
def in_degrees(v: float) -> float:
    return v * pi / 180


@jit(nopython=True, nogil=True)
def to_degrees(v: float) -> float:
    return v / pi * 180


@jit(nopython=True, nogil=True)
def in_nanosec(v: float) -> float:
    return v * 1e-9 * HARTREE / HBAR


@jit(nopython=True, nogil=True)
def to_nanosec(v: float) -> float:
    return v / 1e-9 / HARTREE * HBAR


@jit(nopython=True, nogil=True)
def to_picosec(v: float) -> float:
    return v / 1e-12 / HARTREE * HBAR


@jit(nopython=True, nogil=True)
def in_picosec(v: float) -> float:
    return v * 1e-12 * HARTREE / HBAR


@jit(nopython=True, nogil=True)
def to_femtosec(v: float) -> float:
    return v / 1e-15 / HARTREE * HBAR


@jit(nopython=True, nogil=True)
def in_femtosec(v: float) -> float:
    return v * 1e-15 * HARTREE / HBAR


@jit(nopython=True, nogil=True)
def to_attosec(v: float) -> float:
    return v / 1e-18 / HARTREE * HBAR


@jit(nopython=True, nogil=True)
def in_attosec(v: float) -> float:
    return v * 1e-18 * HARTREE / HBAR


@jit(nopython=True, nogil=True)
def in_centimetres(v: float) -> float:
    return v * 1e-2 / BOHR


@jit(nopython=True, nogil=True)
def to_centimetres(v: float) -> float:
    return v / 1e-2 * BOHR


@jit(nopython=True, nogil=True)
def in_millimetres(v: float) -> float:
    return v * 1e-3 / BOHR


@jit(nopython=True, nogil=True)
def to_millimetres(v: float) -> float:
    return v / 1e-3 * BOHR


@jit(nopython=True, nogil=True)
def in_volts(v: float) -> float:
    return v * E / HARTREE


@jit(nopython=True, nogil=True)
def to_volts(v: float) -> float:
    return v / E * HARTREE


@jit(nopython=True, nogil=True)
def in_gausses(v: float) -> float:
    return v * 1e-4 * E * BOHR ** 2 / HBAR


@jit(nopython=True, nogil=True)
def to_gausses(v: float) -> float:
    return v / 1e-4 / E / BOHR ** 2 * HBAR


in_electronvolts = in_volts
to_electronvolts = to_volts


@jit(nopython=True, nogil=True)
def in_joules(v: float) -> float:
    return v / HARTREE


@jit(nopython=True, nogil=True)
def to_joules(v: float) -> float:
    return v * HARTREE


@jit(nopython=True, nogil=True)
def in_joules_per_square_centimetre(v: float) -> float:
    return v / HARTREE / (1e-2 / BOHR)**2


@jit(nopython=True, nogil=True)
def to_joules_per_square_centimetre(v: float) -> float:
    return v * HARTREE * (1e-2 / BOHR)**2


@jit(nopython=True, nogil=True)
def in_daltons(v: float) -> float:
    return v * MA / ME


@jit(nopython=True, nogil=True)
def to_daltons(v: float) -> float:
    return v / MA * ME


@jit(nopython=True, nogil=True)
def identity(v: float) -> float:
    return v


f = {
    "au": identity,
    "rad": identity,
    "deg": in_degrees,
    "ns": in_nanosec,
    "ps": in_picosec,
    "fs": in_femtosec,
    "as": in_attosec,
    "mm": in_millimetres,
    "V": in_volts,
    "G": in_gausses,
    "eV": in_electronvolts,
    "J/cm2": in_joules_per_square_centimetre,
    "Da": in_daltons,
    "u": in_daltons,
}
g = {
    "au": identity,
    "rad": identity,
    "deg": to_degrees,
    "ns": to_nanosec,
    "ps": to_picosec,
    "fs": to_femtosec,
    "as": to_attosec,
    "mm": to_millimetres,
    "V": to_volts,
    "G": to_gausses,
    "eV": to_electronvolts,
    "J/cm2": to_joules_per_square_centimetre,
    "Da": to_daltons,
    "u": to_daltons,
}


def convert_units(v: float, fr: str = "au", to: str = "au") -> float:
    if fr not in f:
        raise ValueError("Unit '{}' is not supported!".format(fr))
    if to not in g:
        raise ValueError("Unit '{}' is not supported!".format(to))
    return g[to](f[fr](v))
