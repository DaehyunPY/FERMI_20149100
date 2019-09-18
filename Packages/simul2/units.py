from typing import Optional, List, Mapping, Any
from importlib.resources import path

from .pint import UnitRegistry

from . import rsc

__all__ = ["Q_", "quantity"]

with path(rsc, "default.txt") as fn:
    ureg = UnitRegistry(str(fn), on_redefinition="raise")

Q_ = ureg.Quantity


def quantity(input: Any,
             fr: Optional[str] = None,
             fr_default: str = "",
             to: Optional[str] = None,
             context: Optional[str] = None) -> Any:
    if isinstance(input, Q_):
        if fr is not None:
            raise ValueError("Arg input already declared its units!")
        q = input
    else:
        if (fr is None
                and isinstance(input, tuple)
                and len(input) == 2
                and isinstance(input[1], str)):
            input, fr = input
        q = Q_(input)
        if q.unitless:
            q *= Q_(1, fr_default if fr is None else fr)
        elif fr is not None:
            raise ValueError("Arg input already declared its units!")
    return q.to(fr_default if to is None else to,
                *[arg for arg in [context] if arg is not None]).m
