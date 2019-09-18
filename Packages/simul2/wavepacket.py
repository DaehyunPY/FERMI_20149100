"""A simple simulation of wave packet.

Refer the details to the journal paper: PRA 45, 4734 (1992).
"""
from importlib.resources import path

import numpy as np
import pandas as pd
import xarray as xr

from . import rsc
from .electricfield import ElectricField

__all__ = ["predefined_target", "WavePacket"]


def predefined_target(name: str) -> pd.DataFrame:
    with path(rsc, "{}.xlsx".format(name)) as fn:
        return pd.read_excel(fn, "Levels")


class WavePacket:
    def __init__(self, field: ElectricField, target: (str, pd.DataFrame)):
        if isinstance(target, str):
            target = predefined_target(target)

        if "config" in target:
            if not target["config"].is_unique:
                raise ValueError(
                    "Values in target['config'] should be unique.")
            idx = target["config"]
        else:
            idx = range(len(target))

        self.__status = pd.DataFrame({
            "config": idx,
            "freq": target["level"],
            "coeff": target["strength"]**0.5 * field.at_k(target["level"]),
        }).set_index("config")

    @property
    def status(self) -> pd.DataFrame:
        return self.__status

    def __call__(self, t: np.ndarray) -> xr.DataArray:
        n = self.__status.index  # dims: [n]
        k = self.__status["freq"]  # dims: [n]
        c = self.__status["coeff"]  # dims: [n]
        a = -1j * np.exp(-1j * k[None, :] * t[:, None]) * c[None, :].conj()
        # dims: [t, n]
        return xr.DataArray(
            (a[:, :, None] * a[:, None, :].conj()).real,
            coords=[t, n, n],
            dims=["t", "n", "n'"],
        )
