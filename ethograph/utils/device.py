"""Torch device resolution (CUDA → MPS → CPU, never hardcoded) and build probes."""

from __future__ import annotations


def resolve_device(preferred: str | None = None) -> str:
    """Pick the best available torch device, honouring *preferred* when usable."""
    try:
        import torch
    except ImportError:
        return "cpu"

    available = ["cpu"]
    if torch.cuda.is_available():
        available.insert(0, "cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        available.insert(0 if "cuda" not in available else 1, "mps")

    if preferred:
        base = preferred.split(":")[0]
        if base in available:
            return preferred
    return available[0]


#: The one install line that repairs a torch/torchvision pair from mixed indexes.
#: The CPU wheel carries the same version number as the CUDA one, so without a
#: forced reinstall the resolver reports both as satisfied and stops; the
#: per-package form leaves their dependencies (numpy!) alone.
TORCH_PAIR_INSTALL = (
    "uv pip install --torch-backend=auto --reinstall-package torch --reinstall-package torchvision torch torchvision"
)


def torchvision_ops_error(device: str) -> str | None:
    """Why torchvision's compiled ops (NMS, RoI align) cannot run on *device*, or ``None``.

    Runs the op itself rather than comparing version strings: a CPU-only
    torchvision wheel beside a CUDA torch imports and version-matches fine
    and only fails at the first GPU NMS — deep inside a YOLO run.
    """
    if device.split(":")[0] == "cpu":
        return None
    try:
        import torch
        import torchvision
    except ImportError as e:
        return f"{e}. Install the pair: {TORCH_PAIR_INSTALL}"
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    try:
        torchvision.ops.nms(boxes, scores, 0.5)
    except (NotImplementedError, RuntimeError) as e:
        first = str(e).split(". ")[0]  # torch's dispatcher error runs to a paragraph
        return (
            f"torchvision {torchvision.__version__} has no {device} kernels (torch {torch.__version__}): {first}. "
            f"Both must come from the same index — {TORCH_PAIR_INSTALL}"
        )
    return None


def require_torchvision_ops(device: str) -> None:
    """Raise before a GPU feature that needs torchvision's ops starts on a build without them."""
    error = torchvision_ops_error(device)
    if error is not None:
        raise RuntimeError(error)
