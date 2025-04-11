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
    mask_tensor_with_boxes,
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
        targets_logits_seg: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks, anchor_to_gt_idxs = self.matcher(
            targets, (predicts_cls.detach(), predicts_box.detach())
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
                targets_logits_seg,
                valid_masks,
                anchor_to_gt_idxs,
                targets,
                cls_norm,
                box_norm,
            )

        return loss_iou, loss_dfl, loss_cls, loss_seg


class MaskLoss(nn.Module):
    """
    Computes the mask loss by comparing predicted masks with the corresponding downsampled
    ground truth masks. The process includes:
      - Downsampling target masks to match the prototype shape.
      - Adjusting ground truth boxes and reshaping them.
      - Computing predicted masks (with sigmoid activation).
      - Masking the predicted masks using the ground truth boxes.
      - Computing a per-mask binary cross entropy loss and aggregating normalized losses.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_logits,
        targets,
        valid_masks,
        anchor_to_gt_idxs,
        gt_boxes,
        cls_norm,
        box_norm,
    ):
        """
        Computes the mask loss based on the predicted logits and ground truth masks.
        
        Args:
            pred_logits (List[Tensor]): List of predicted logits for each prototype 
                List of predicted masks + prototype, the shapes are (Batch size, n_prototype_masks, h, w) and they are normally in decreasing dimension except the last one that is the prototype and is normally bigger.
                Eg: [Tensor(3, 32, 64, 64), Tensor(3, 32, 32, 32), Tensor(3, 32, 16, 16), Tensor(3, 32, 128, 128)]
            targets (Tensor): Ground truth masks of shape (Batch size, max_preds_in_batch, h, w).
                Eg.: Tensor(3, 10, 512, 512)
            valid_masks (bool Tensor): Boolean tensor indicating valid masks of shape (Batch size, all_predicted_anchors).
                Eg.: Tensor(3, 5376)
            anchor_to_gt_idxs (Tensor): Tensor mapping anchors to ground truth indices of shape (Batch size, all_predicted_anchors, 1).
                Eg.: Tensor(3, 5376, 1)
            gt_boxes (Tensor): Ground truth boxes of shape (Batch size, max_preds_in_batch, 5).
                Eg.: Tensor(3, 10, 5)
            cls_norm (Tensor): Normalization factor for the classification loss.
                Eg.: Tensor(1.0)
            box_norm (Tensor): Normalization factor for the box loss. The shape is (Number of valid masks).
                Eg.: Tensor(valid_masks.sum())
        """
        # Determine prototype dimensions from the last predicted tensor.
        proto_h, proto_w = pred_logits[-1].shape[-2:]

        # Downsample target masks to match the prototype dimensions, then binarize.
        downsampled_targets = (
            F.interpolate(
                targets,
                size=(proto_h, proto_w),
                mode="bilinear",
                align_corners=False,
            )
            .gt(0.5)
            .float()
        )

        # Remove the first coordinate from gt_boxes and rescale to the downsampled target shape.
        gt_boxes = gt_boxes[..., 1:]
        rescaled_gt_boxes = reshape_batched_bboxes(
            gt_boxes, targets.shape[-2:], downsampled_targets.shape[-2:]
        )

        # Obtain predicted masks (apply sigmoid activation).
        preds = get_mask_preds(pred_logits, sigmoid=True)

        # Initialize a tensor to accumulate losses for all valid predictions.
        total_losses = torch.zeros(valid_masks.sum(), device=pred_logits[0].device)
        loss_index = 0

        # Iterate over each sample in the batch.
        for down_target, pred, valid_mask, anchor_to_gt, gt_box in zip(
            downsampled_targets,
            preds,
            valid_masks,
            anchor_to_gt_idxs,
            rescaled_gt_boxes,
        ):
            num_matches = valid_mask.sum()
            if num_matches == 0:
                continue

            # Select predictions and corresponding ground truth indices based on the valid mask.
            valid_pred = pred[valid_mask]
            valid_anchor_to_gt = anchor_to_gt[valid_mask]

            # Retrieve the matched ground truth boxes and masks for the valid predictions.
            # Shape of matched_gt_boxes: [num_matches, 4]
            matched_gt_boxes = gt_box[valid_anchor_to_gt][:, 0, :]
            # Shape of matched_gt_masks: [num_matches, proto_h, proto_w]
            matched_gt_masks = down_target[valid_anchor_to_gt][:, 0, :, :]

            # Mask the predicted masks using the ground truth boxes.
            masked_preds = mask_tensor_with_boxes(valid_pred, matched_gt_boxes)

            # Compute per-pixel binary cross entropy loss.
            loss_per_pixel = F.binary_cross_entropy(
                masked_preds, matched_gt_masks, reduction="none"
            )
            # Sum the loss for each mask.
            loss_per_mask = loss_per_pixel.view(loss_per_pixel.shape[0], -1).sum(dim=-1)

            # Compute the area of each ground truth mask.
            area_per_mask = matched_gt_masks.view(matched_gt_masks.shape[0], -1).sum(
                dim=-1
            )
            norm_loss_per_mask = loss_per_mask / (area_per_mask.float() + 1e-6)

            total_losses[loss_index : loss_index + num_matches] = norm_loss_per_mask
            loss_index += num_matches

        # Aggregate the loss across all valid predictions and apply normalizations:
        # - box_norm and cls_norm (like the other losses)
        # - valid_masks.sum(): since the mask loss was 3 orders of magnitude higher than the others, this makes them comparable
        final_loss = (total_losses * box_norm).sum() / (cls_norm * valid_masks.sum())
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
            ), "When computing the loss with the segmentation masks, you must provide the seg targets and pred logits."

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
