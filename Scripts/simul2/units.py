from importlib.resources import path

from pint import UnitRegistry

from . import rsc

__all__ = ["Q_"]

with path(rsc, "default.txt") as fn:
    ureg = UnitRegistry(str(fn))

Q_ = ureg.Quantity
