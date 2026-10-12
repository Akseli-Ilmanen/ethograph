"""SAMURAI's motion-aware tracking on top of OCTRON's SAM2 predictor.

SAMURAI (Yang et al., 2024, https://github.com/yangchris11/samurai) changes
SAM 2 in two places, both in ``SAM2Base``, without retraining:

1. **Mask selection.** SAM proposes three masks per frame and keeps the one
   with the highest predicted IoU. SAMURAI runs a Kalman filter over the
   object's box and keeps the mask that best agrees with *both* SAM's IoU and
   the filter's predicted box. That is what stops the track jumping to a
   look-alike elsewhere in the frame.
2. **Memory bank.** SAM attends to the last few frames whatever happened in
   them. SAMURAI only lets frames whose mask scored well (IoU, object score,
   motion agreement) into the memory, so a bad frame does not poison the next.

Upstream ships these as a replacement ``sam2`` package. Here they are a
subclass of OCTRON's predictor, so the installed ``sam2`` and OCTRON stay as
they are; only the Kalman filter is copied (``ethograph/_vendor/samurai``).
Two deviations, both forced by OCTRON: the memory-bank selection reads the
frames OCTRON has kept (it drops outputs older than 16 frames; upstream keeps
them all), and object pointers are drawn from the same selected frames.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from octron.sam_octron.helpers.sam2_octron import SAM2_octron
from sam2.modeling.sam2_base import NO_OBJ_SCORE

from ethograph._vendor.samurai.kalman_filter import KalmanFilter
from ethograph.utils.device import resolve_device

logger = logging.getLogger(__name__)

_AUTOCAST_ENTERED = False


@dataclass(frozen=True)
class SamuraiParams:
    """SAMURAI's hyperparameters, defaulting to its published ``sam2.1_hiera_t.yaml``."""

    #: Frames of agreement before the Kalman filter is trusted to re-rank masks.
    stable_frames_threshold: int = 15
    #: A frame whose best mask IoU falls below this resets the filter's trust.
    stable_ious_threshold: float = 0.3
    #: Object-score logit below which SAM reports "no object" (upstream SAM 2 uses 0).
    min_obj_score_logits: float = -1.0
    #: Weight of the filter's box IoU against SAM's own IoU when re-ranking.
    kf_score_weight: float = 0.25
    #: A past frame enters the memory bank only above these three scores.
    memory_bank_iou_threshold: float = 0.5
    memory_bank_obj_score_threshold: float = 0.0
    memory_bank_kf_score_threshold: float = 0.0

    def hydra_overrides(self) -> list[str]:
        return [f"++model.{key}={value}" for key, value in asdict(self).items()]


def select_memory_frames(
    outputs: dict[int, dict],
    frame_idx: int,
    params: SamuraiParams,
    max_frames: int,
) -> list[int]:
    """The past frames allowed into memory for *frame_idx*, oldest first.

    Walks back from ``frame_idx - 1`` over the frames in *outputs*, keeping
    those whose stored ``best_iou_score``, ``object_score_logits`` and
    ``kf_score`` clear the thresholds, up to ``max_frames - 1`` of them. The
    frame immediately before is always included when present, as upstream
    does. A frame with no stored scores (a prompted frame) passes.
    """
    valid: list[int] = []
    for i in sorted((k for k in outputs if k < frame_idx), reverse=True):
        out = outputs[i]
        iou, obj, kf = out.get("best_iou_score"), out.get("object_score_logits"), out.get("kf_score")
        passes = (
            (iou is None or float(iou) > params.memory_bank_iou_threshold)
            and (obj is None or float(obj) > params.memory_bank_obj_score_threshold)
            and (kf is None or float(kf) > params.memory_bank_kf_score_threshold)
        )
        if passes:
            valid.insert(0, i)
        if len(valid) >= max_frames - 1:
            break
    if frame_idx - 1 in outputs and frame_idx - 1 not in valid:
        valid.append(frame_idx - 1)
    return valid


def _box_of(mask: torch.Tensor) -> list[float]:
    """``[x1, y1, x2, y2]`` of the positive pixels of a ``(H, W)`` mask, zeros when empty."""
    nonzero = torch.argwhere(mask > 0.0)
    if len(nonzero) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    y_min, x_min = nonzero.min(dim=0).values
    y_max, x_max = nonzero.max(dim=0).values
    return [x_min.item(), y_min.item(), x_max.item(), y_max.item()]


class SamuraiOctron(SAM2_octron):
    """OCTRON's predictor with SAMURAI's mask selection and memory bank. Tracks one object."""

    def __init__(self, **kwargs):
        names = SamuraiParams.__dataclass_fields__
        self.params = SamuraiParams(**{k: kwargs.pop(k) for k in list(kwargs) if k in names})
        self.samurai_mode = True
        self.kf = KalmanFilter()
        self.reset_motion()
        super().__init__(**kwargs)
        if self.memory_temporal_stride_for_eval != 1:
            raise ValueError("SAMURAI's memory bank assumes memory_temporal_stride_for_eval == 1")

    def reset_motion(self) -> None:
        """Forget the Kalman filter's state; call it whenever tracking restarts from a new prompt."""
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0
        self._last_best_iou: torch.Tensor | None = None
        self._last_kf_iou: torch.Tensor | None = None

    # -- 1. mask selection ------------------------------------------------------------

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """Upstream SAM 2's heads, with SAMURAI's Kalman re-ranking of the three candidate masks."""
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.params.min_obj_score_logits
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        kf_ious: torch.Tensor | None = None
        if multimask_output:
            batch_inds = torch.arange(B, device=device)
            best_iou_inds = torch.argmax(ious, dim=-1)
            if self.samurai_mode and B == 1:
                best_iou_inds, kf_ious = self._samurai_select(ious, high_res_multimasks, best_iou_inds, device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
            self._last_best_iou = ious[0][best_iou_inds[0]]
            self._last_kf_iou = kf_ious[best_iou_inds[0]] if kf_ious is not None else None
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
            self._last_best_iou = ious[0][0]
            self._last_kf_iou = None

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _samurai_select(
        self,
        ious: torch.Tensor,
        high_res_multimasks: torch.Tensor,
        best_iou_inds: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """SAMURAI's three regimes: start the filter, warm it up on SAM's choice, then let it vote."""
        p = self.params
        if self.kf_mean is None or self.stable_frames == 0:
            box = _box_of(high_res_multimasks[0, best_iou_inds[0]])
            self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(box))
            self.stable_frames += 1
            return best_iou_inds, None
        self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
        if self.stable_frames < p.stable_frames_threshold:
            box = _box_of(high_res_multimasks[0, best_iou_inds[0]])
            if ious[0][best_iou_inds[0]] > p.stable_ious_threshold:
                self.kf_mean, self.kf_covariance = self.kf.update(
                    self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(box)
                )
                self.stable_frames += 1
            else:
                self.stable_frames = 0
            return best_iou_inds, None
        boxes = [_box_of(high_res_multimasks[0, i]) for i in range(ious.shape[1])]
        kf_ious = torch.tensor(self.kf.compute_iou(self.kf_mean[:4], boxes), device=device)
        weighted = p.kf_score_weight * kf_ious + (1 - p.kf_score_weight) * ious
        best_iou_inds = torch.argmax(weighted, dim=-1)
        if ious[0][best_iou_inds[0]] < p.stable_ious_threshold:
            self.stable_frames = 0
        else:
            chosen = boxes[int(best_iou_inds[0])]
            self.kf_mean, self.kf_covariance = self.kf.update(
                self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(chosen)
            )
        return best_iou_inds, kf_ious

    def track_step(self, *args, **kwargs):
        """Upstream's step, plus the scores the memory bank reads back on later frames."""
        current_out = super().track_step(*args, **kwargs)
        current_out["best_iou_score"] = self._last_best_iou
        current_out["kf_score"] = self._last_kf_iou
        return current_out

    # -- 2. memory bank ------------------------------------------------------------

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """Upstream's fusion, fed only the past frames SAMURAI admits to memory.

        Upstream looks memories up by offset (``frame_idx - 1``, ``- 2`` ...).
        The admitted frames are re-keyed onto those offsets, most recent first,
        for the duration of the call, so upstream's temporal encoding by slot
        is kept exactly as SAMURAI assigns it.
        """
        if not self.samurai_mode or is_init_cond_frame or track_in_reverse:
            return super()._prepare_memory_conditioned_features(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                track_in_reverse,
            )
        original = output_dict["non_cond_frame_outputs"]
        valid = select_memory_frames(original, frame_idx, self.params, self.max_obj_ptrs_in_encoder)
        output_dict["non_cond_frame_outputs"] = {frame_idx - 1 - j: original[v] for j, v in enumerate(reversed(valid))}
        try:
            return super()._prepare_memory_conditioned_features(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                track_in_reverse,
            )
        finally:
            output_dict["non_cond_frame_outputs"] = original


def build_samurai(config_name: str, checkpoint: Path, params: SamuraiParams) -> tuple[SamuraiOctron, torch.device]:
    """OCTRON's ``build_sam2_octron`` with :class:`SamuraiOctron` as the model class.

    OCTRON's builder hard-codes its own class as the hydra target, so the
    build is repeated here: the same bundled SAM 2.1 configs, the same
    checkpoint loader, the same bfloat16 autocast on CUDA.
    """
    import octron.sam_octron.helpers.build_sam2_octron as octron_build
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.build_sam import _load_checkpoint

    global _AUTOCAST_ENTERED
    device = torch.device(resolve_device())
    if device.type == "cuda" and not _AUTOCAST_ENTERED:
        # Entered once per process, as OCTRON does for its single model. Entering it per
        # build would stack contexts whose cast-weight caches keep every model ever built.
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        _AUTOCAST_ENTERED = True
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    config_dir = Path(octron_build.__file__).resolve().parent.parent / "configs" / "sam2.1"
    overrides = [f"++model._target_={SamuraiOctron.__module__}.{SamuraiOctron.__name__}", *params.hydra_overrides()]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, str(checkpoint))
    model = model.to(device)
    model.eval()
    logger.info("SAMURAI (%s) built on %s", config_name, device)
    return model, device
