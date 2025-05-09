from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from yolo.config.config import Config, LossConfig
from yolo.utils.bounding_box_utils import (
    BoxMatcher,
    Vec2Box,
    calculate_iou,
    get_tensor_mask_from_boxes,
    reshape_batched_bboxes,
)
from yolo.utils.logger import logger
from yolo.utils.model_utils import get_mask_preds


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 assing pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(
        self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor
    ) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        predicts_bbox: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self,
        predicts_anc: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(
            ((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1
        ).clamp(0, self.reg_max - 1.01)
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = (
            label_right - picked_targets,
            picked_targets - label_left,
        )

        loss_left = F.cross_entropy(
            picked_predict, label_left.to(torch.long), reduction="none"
        )
        loss_right = F.cross_entropy(
            picked_predict, label_right.to(torch.long), reduction="none"
        )
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


class YOLOLoss:
    def __init__(
        self,
        loss_cfg: LossConfig,
        vec2box: Vec2Box,
        class_num: int = 80,
        reg_max: int = 16,
        seg=False,
    ) -> None:
        self.class_num = class_num
        self.vec2box = vec2box

        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()

        self.seg = None
        if seg is not None:
            self.seg = MaskLoss()

        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box, reg_max)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(
        self,
        predicts: List[Tensor],
        targets: Tensor,
        seg_logits_preds: Optional[List[Tensor]] = None,
        targets_seg: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks, aligned_masks_unique_idxs = self.matcher(
            targets, (predicts_cls.detach(), predicts_box.detach()), targets_seg
        )

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- SEG -- ##
        loss_seg = 0
        if self.seg is not None and seg_logits_preds is not None:
            loss_seg = self.seg(
                seg_logits_preds,
                targets_seg,
                aligned_masks_unique_idxs,
                valid_masks,
                align_targets.clone()[..., 2:],
                cls_norm,
                box_norm,
            )

        return loss_iou, loss_dfl, loss_cls, loss_seg


class MaskLoss(nn.Module):
    """
    Computes the mask loss by comparing predicted masks with the downsampled ground truth masks.
    The steps include:
      - Downsampling ground truth masks to prototype dimensions.
      - Rescaling the ground truth boxes.
      - Generating predicted masks with a sigmoid activation.
      - Gathering the aligned mask regions and applying ground truth boxes.
      - Computing a normalized binary cross entropy loss for each mask.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        seg_logits_list,
        gt_masks,
        unique_mask_indices,
        valid_mask_flags,
        gt_boxes_aligned,
        cls_normalizer,
        box_normalizer,
    ):
        """
        Computes the mask loss.

        Args:
            seg_logits_list (List[Tensor]): List of predicted logits for each prototype.
                Example shapes: [Tensor(3, 32, 64, 64), Tensor(3, 32, 32, 32), ...]
            gt_masks (Tensor): Ground truth masks with shape (Batch, max_preds, H, W).
                Example: Tensor(3, 10, 512, 512)
            unique_mask_indices (List[Tensor]): Unique indices to gather aligned masks for each batch item.
            valid_mask_flags (Tensor): Boolean tensor indicating valid masks; shape (Batch, total_anchors).
            gt_boxes_aligned (Tensor): Ground truth boxes, shape (Batch, max_preds, 5).
            cls_normalizer (Tensor): Normalizing factor for classification loss.
            box_normalizer (Tensor): Normalizing factor for box loss, shaped as (Number of valid masks).

        Returns:
            final_loss (Tensor): The aggregated normalized mask loss.
        """
        # Determine target prototype dimensions from the last predicted logits.
        proto_h, proto_w = seg_logits_list[-1].shape[-2:]

        # Downsample ground truth masks to match prototype shape and binarize.
        downsampled_gt_masks = (
            F.interpolate(
                gt_masks, size=(proto_h, proto_w), mode="bilinear", align_corners=False
            )
            .gt(0.5)
            .float()
        )

        # Rescale ground truth boxes to match the downsampled mask dimensions.
        rescaled_gt_boxes = reshape_batched_bboxes(
            gt_boxes_aligned, gt_masks.shape[-2:], downsampled_gt_masks.shape[-2:]
        )

        # Generate predicted masks by applying sigmoid activation.
        pred_masks_logits = get_mask_preds(seg_logits_list, sigmoid=False)

        # Initialize a tensor to accumulate the normalized losses.
        total_losses = torch.zeros(
            valid_mask_flags.sum(), device=seg_logits_list[0].device
        )
        loss_count = 0

        # Process each item in the batch.
        for down_gt, pred_mask_logits, valid_flags, boxes, indices in zip(
            downsampled_gt_masks,
            pred_masks_logits,
            valid_mask_flags,
            rescaled_gt_boxes,
            unique_mask_indices,
        ):
            num_valid = valid_flags.sum()
            if num_valid == 0:
                continue

            # Gather the aligned ground truth masks based on unique indices.
            valid_indices = indices[valid_flags]
            valid_gt_masks = torch.gather(
                down_gt, 0, valid_indices.repeat(1, *down_gt.shape[1:])
            )

            # Select only valid predictions and corresponding ground truth elements.
            valid_pred_masks_logits = pred_mask_logits[valid_flags]
            valid_boxes = boxes[valid_flags]

            # Apply ground truth boxes to mask the predictions.
            mask = get_tensor_mask_from_boxes(valid_pred_masks_logits, valid_boxes)

            # Compute per-pixel binary cross entropy loss without reduction.
            pixel_loss = F.binary_cross_entropy_with_logits(
                valid_pred_masks_logits, valid_gt_masks, reduction="none"
            )

            pixel_loss = pixel_loss * mask

            # Sum loss per mask.
            loss_per_mask = pixel_loss.view(pixel_loss.size(0), -1).sum(dim=-1)

            # Compute the area (sum of pixels) of each ground truth mask.
            area_per_mask = valid_gt_masks.view(valid_gt_masks.size(0), -1).sum(dim=-1)
            # Normalize loss per mask by its area.
            norm_loss_per_mask = loss_per_mask / (area_per_mask.float() + 1e-6)

            total_losses[loss_count : loss_count + num_valid] = norm_loss_per_mask
            loss_count += num_valid

        # Aggregate the normalized loss with provided normalizers.
        final_loss = (total_losses * box_normalizer).sum() / (
            cls_normalizer * valid_mask_flags.sum()
        )
        return final_loss


class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]
        self.seg_rate = loss_cfg.objective.get("LincombMaskLoss", None)

        self.loss = YOLOLoss(
            loss_cfg,
            vec2box,
            class_num=cfg.dataset.class_num,
            reg_max=cfg.model.anchor.reg_max,
            seg=self.seg_rate is not None,
        )

    def __call__(
        self,
        aux_predicts: List[Tensor],
        main_predicts: List[Tensor],
        targets: Tensor,
        # only for the segmentation loss
        target_seg: Optional[Tensor] = None,
        aux_seg_logits: Optional[List[Tensor]] = None,
        main_seg_logits: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if self.seg_rate is not None:
            assert (
                target_seg is not None
                and aux_seg_logits is not None
                and main_seg_logits is not None
            ), (
                "When computing the loss with the segmentation masks, you must provide the seg targets and pred logits."
            )

        # TODO: Need Refactor this region, make it flexible!
        aux_iou, aux_dfl, aux_cls, aux_seg = self.loss(
            aux_predicts, targets, aux_seg_logits, target_seg
        )
        main_iou, main_dfl, main_cls, main_seg = self.loss(
            main_predicts, targets, main_seg_logits, target_seg
        )

        self.seg_rate = self.seg_rate or torch.tensor(0.0)

        total_loss = [
            self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            self.cls_rate * (aux_cls * self.aux_rate + main_cls),
            self.seg_rate * (aux_seg * self.aux_rate + main_seg),
        ]
        loss_dict = {
            f"Loss/{name}Loss": value.detach().item()
            for name, value in zip(["Box", "DFL", "BCE", "LincombMask"], total_loss)
        }
        return sum(total_loss), loss_dict


def create_loss_function(cfg: Config, vec2box) -> DualLoss:
    # TODO: make it flexible, if cfg doesn't contain aux, only use SingleLoss
    loss_function = DualLoss(cfg, vec2box)
    logger.info(":white_check_mark: Success load loss function")
    return loss_function
