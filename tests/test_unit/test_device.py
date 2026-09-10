"""The torchvision build probe: run the op, never compare version strings."""

from __future__ import annotations

import pytest

from ethograph.utils.device import TORCH_PAIR_INSTALL, require_torchvision_ops, torchvision_ops_error

pytest.importorskip("torchvision")


def test_cpu_always_has_the_ops():
    assert torchvision_ops_error("cpu") is None
    require_torchvision_ops("cpu:0")


def test_a_device_without_kernels_names_the_repair():
    """A device torch cannot run at all fails the same way a CPU-only wheel does on CUDA."""
    error = torchvision_ops_error("meta")
    assert error is not None and TORCH_PAIR_INSTALL in error
    with pytest.raises(RuntimeError, match="torchvision"):
        require_torchvision_ops("meta")
