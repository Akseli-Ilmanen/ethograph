"""Driving the vendored E2E-Spot (``ethograph/spot/e2espot/``, see its ``NOTICE.md``).

Upstream's training loop is run as a subprocess rather than imported: it owns
``argparse``, a global config dict and its own logging, and rewriting it is
exactly what vendoring is meant to avoid. What this module adds is what a
subprocess cannot do for itself — resume after a crash, and mirror the output
to a log beside the run.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: Seconds to wait before resuming after a crash — long enough for the CUDA
#: context of the dead process to be released.
RETRY_PAUSE_S = 20

#: The vendored tree, in upstream's layout.
E2ESPOT_DIR = Path(__file__).resolve().parent / "e2espot"


def script_command(script: str) -> list[str]:
    """``python -m`` for one of upstream's scripts (``train_e2e`` / ``test_e2e``), run from any directory."""
    return [sys.executable, "-m", f"ethograph.spot.e2espot.{script}"]


def has_checkpoint(save_dir: Path) -> bool:
    """Whether ``--resume`` has an epoch to pick up from."""
    return save_dir.is_dir() and any(save_dir.glob("optim_*.pt"))


def run_logged(command: list[str], log_path: Path, cwd: Path) -> int:
    """Run *command*, mirroring its output to the console and to *log_path*.

    Chunked rather than line-based so tqdm's ``\\r`` updates reach the log as
    they happen. ``expandable_segments`` lets the CUDA allocator grow in place
    instead of failing on fragmentation — the failure a run near the card's
    limit otherwise hits hours in, during validation.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_path.open("ab") as log,
        subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc,
    ):
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
    return proc.returncode


def run_with_retries(command: list[str], save_dir: Path, retries: int) -> None:
    """Run *command*, resuming from the last checkpoint after a crash.

    ``train_e2e.py`` writes ``checkpoint_NNN.pt`` + ``optim_NNN.pt`` after
    every epoch and ``--resume`` continues from the newest, so a failure costs
    at most one epoch. Every attempt's output lands in ``{save_dir}/train.log``.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train.log"
    for attempt in range(retries + 1):
        attempt_command = list(command)
        if attempt and has_checkpoint(save_dir):
            attempt_command.append("--resume")
        logger.info("Attempt %d/%d: %s", attempt + 1, retries + 1, " ".join(attempt_command))
        code = run_logged(attempt_command, log_path, cwd=save_dir)
        if code == 0:
            return
        logger.error("train_e2e.py exited with %d (see %s)", code, log_path)
        if attempt == retries:
            raise RuntimeError(f"train_e2e.py failed with exit code {code}; see {log_path}")
        logger.info("Restarting in %d s, resuming from the last checkpoint", RETRY_PAUSE_S)
        time.sleep(RETRY_PAUSE_S)


#: What each suffix of an architecture name means, for :func:`describe_architectures`.
TEMPORAL_MODULES = {
    "": "no temporal mixing in the backbone (the GRU head alone)",
    "tsm": "Temporal Shift Module: a fixed slice of channels shifted +-1 frame",
    "gsm": "Gate Shift Module: learned gates decide what shifts +-1 frame (E2E-Spot's own)",
    "msagsm": "Multi-scale gated shift: GSM at several reaches at once, behind grouped attention "
    "(ethograph.spot.msagsm; reach set by model.shift_scales_ms)",
}

BACKBONES = {
    "rn18": "ResNet-18 (torchvision)",
    "rn50": "ResNet-50 (torchvision)",
    "rny002": "RegNetY-200MF (timm)",
    "rny008": "RegNetY-800MF (timm) - E2E-Spot's default",
    "convnextt": "ConvNeXt-Tiny (timm)",
}

_CHOICES_RE = re.compile(r"--feature_arch'.*?choices=\[(.*?)\]", re.S)


def feature_architectures(root: Path | None = None) -> list[str]:
    """Every ``--feature_arch`` the vendored trainer accepts, read off its CLI.

    The vendored ``train_e2e.py`` is the one authority on what can be trained;
    reading its ``choices=[...]`` keeps this list from drifting when it gains
    an architecture. No import — the trainer pulls in torch and timm.
    """
    source = (root or E2ESPOT_DIR) / "train_e2e.py"
    match = _CHOICES_RE.search(source.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{source}: could not find the --feature_arch choices")
    return re.findall(r"'([^']+)'", match.group(1))


def describe_architecture(name: str) -> str:
    """``backbone + temporal module`` in words, for one architecture name."""
    backbone, _, module = name.partition("_")
    return f"{BACKBONES.get(backbone, backbone)} + {TEMPORAL_MODULES.get(module, module)}"


#: What a 200-frame loader batch of a RegNetY-008 + GSM needs, measured on a
#: 10 GB card (`scripts/spot_point_events.md`, "the card must be empty").
GB_PER_200_FRAMES = 6.0


def gpu_holders() -> str:
    """Who holds the card's memory, from nvidia-smi — the one thing a user can act on."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "(nvidia-smi not available)"
    return out.stdout.strip() or "(nvidia-smi lists no process)"


def frame_budget() -> int:
    """Frames per loader batch the card present holds: the 10 GB measurement scaled by its memory.

    :data:`~ethograph.spot.config.MAX_FRAMES_PER_BATCH` (200) was measured on
    a 10 GB card; a 24 GB card holds ~480, an 8 GB one ~160. No CUDA device
    (the CPU, or a test) reads as the measured card, so a config resolves the
    same on every machine that has none.
    """
    import torch

    from ethograph.spot.config import MAX_FRAMES_PER_BATCH, MIN_CLIP_LEN

    if not torch.cuda.is_available():
        return MAX_FRAMES_PER_BATCH
    _free, total = torch.cuda.mem_get_info()
    return max(MIN_CLIP_LEN, int(MAX_FRAMES_PER_BATCH * (total / 1e9) / 10.0))


def check_vram(frames_per_batch: int) -> None:
    """Refuse to start a run the card cannot hold.

    A run that pages into system RAM does not fail — it runs at 20x slow with
    the card reading 100 % busy, and the model never migrates back once
    memory frees up. So this raises *before* anything is loaded, GUI open or
    not, naming what holds the memory. No CUDA device = nothing to check.
    """
    import torch

    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    needed = GB_PER_200_FRAMES * frames_per_batch / 200.0
    free_gb, total_gb = free / 1e9, total / 1e9
    if free_gb < needed:
        raise RuntimeError(
            f"{free_gb:.1f} GB of {total_gb:.1f} GB free on the GPU, but {frames_per_batch} frames per batch "
            f"need about {needed:.1f} GB. A run that starts anyway pages and crawls rather than failing. "
            f"Holding the rest:\n{gpu_holders()}"
        )
    logger.info("GPU: %.1f of %.1f GB free, %.1f GB needed", free_gb, total_gb, needed)
