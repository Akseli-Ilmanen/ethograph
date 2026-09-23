"""Which xarray engine can reach a NetCDF path on this machine."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

NetcdfEngine = Literal["netcdf4", "h5netcdf"]


def netcdf_engine(path: str | Path) -> NetcdfEngine:
    """The engine every ``.nc`` open and write passes to xarray.

    netCDF-C on Windows reads a path's bytes in the ANSI code page while
    netCDF4-python hands it UTF-8, so a folder named ``präsi`` is "No such
    file or directory". h5py speaks Unicode paths, and both engines write
    the same NetCDF4/HDF5 file.
    """
    if sys.platform == "win32" and not os.path.abspath(path).isascii():
        return "h5netcdf"
    return "netcdf4"
