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
from einops import rearrange


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
        loss_coeffs_diversity = 0
        if self.seg is not None and seg_logits_preds is not None:
            loss_seg, loss_coeffs_diversity = self.seg(
                seg_logits_preds,
                targets_seg,
                aligned_masks_unique_idxs,
                valid_masks,
                align_targets.clone()[..., 2:],
                cls_norm,
                box_norm,
            )

        return loss_iou, loss_dfl, loss_cls, loss_seg, loss_coeffs_diversity


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

    @staticmethod
    def coeff_diversity_loss(coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        """
        num_pos = coeffs.size(0)
        instance_t = instance_t.view(-1)

        coeffs_norm = F.normalize(coeffs, dim=1)
        cos_sim = coeffs_norm @ coeffs_norm.t()

        inst_eq = (
            instance_t[:, None].expand_as(cos_sim)
            == instance_t[None, :].expand_as(cos_sim)
        ).float()

        # Rescale to be between 0 and 1
        cos_sim = (cos_sim + 1) / 2

        # If they're the same instance, use cosine distance, else use cosine similarity
        loss = (1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)

        # Only divide by num_pos once because we're summing over a num_pos x num_pos tensor
        # and all the losses will be divided by num_pos at the end, so just one extra time.
        return loss.sum() / (num_pos**2)

    def forward(
        self,
        prototype_logits: List[Tensor],
        ground_truth_masks: Tensor,
        mask_indices_list: List[Tensor],
        valid_mask_map: Tensor,
        aligned_gt_boxes: Tensor,
        classification_norm: Tensor,
        box_norm: Tensor,
    ):
        """
        Compute the per‐mask BCE loss and the coefficient diversity loss.

        Workflow:
        1. Derive the target mask size from the last prototype.
        2. If there are no GT masks present, return zero loss immediately.
        3. Downsample and binarize all GT masks to prototype resolution.
        4. Rescale GT boxes into the downsampled mask coordinate frame.
        5. Decode predicted mask logits and flatten all mask coefficients.
        6. Iterate over each batch element:
            a. Skip if there are no valid masks.
            b. Extract only the GT masks, predicted logits, boxes and coeffs for valid anchors.
            c. Build a box‐based binary mask to zero out irrelevant pixels.
            d. Compute per‐pixel BCE without reduction, apply the box mask.
            e. Sum losses per mask and normalize by mask area.
            f. Accumulate the normalized losses and the coeff‐diversity term.
        7. Combine all per‐mask losses into the final loss, and average the diversity term.
        """
        # 1) Determine prototype height & width
        proto_h, proto_w = prototype_logits[-1].shape[-2:]

        # 2) Early exit when no ground‐truth masks exist
        if ground_truth_masks.shape[1] == 0:
            return torch.tensor(0.0, device=ground_truth_masks.device), torch.tensor(
                0.0, device=ground_truth_masks.device
            )

        # 3) Downsample & binarize GT masks to (proto_h, proto_w)
        down_gt_masks = (
            F.interpolate(
                ground_truth_masks,
                size=(proto_h, proto_w),
                mode="bilinear",
                align_corners=False,
            )
            .gt(0.5)
            .float()
        )

        # 4) Warp GT boxes into the downsampled mask space
        scaled_gt_boxes = reshape_batched_bboxes(
            aligned_gt_boxes,
            original_shape=ground_truth_masks.shape[-2:],  # (H, W)
            new_shape=down_gt_masks.shape[-2:],  # (proto_h, proto_w)
        )

        # 5) Prepare predictions: raw logits and flattened coefficients
        raw_pred_masks = get_mask_preds(prototype_logits, sigmoid=False)
        coeff_list = []
        for proto in prototype_logits[:-1]:
            # (B, n_coeffs, w, h) → (B, w*h, n_coeffs)
            coeff_list.append(rearrange(proto, "B C H W -> B (H W) C"))
        all_coeffs = torch.cat(coeff_list, dim=1)

        # Allocate storage for each valid mask’s normalized loss
        total_valid = int(valid_mask_map.sum())
        per_mask_losses = torch.zeros(total_valid, device=prototype_logits[0].device)
        loss_cursor = 0
        diversity_accum = torch.zeros(1, device=prototype_logits[0].device)

        # 6) Loop over batch
        for (
            single_down_gt,
            single_pred_logits,
            valid_flags,
            single_boxes,
            mask_indices,
            single_coeffs,
        ) in zip(
            down_gt_masks,
            raw_pred_masks,
            valid_mask_map,
            scaled_gt_boxes,
            mask_indices_list,
            all_coeffs,
        ):
            num_valid = int(valid_flags.sum())
            if num_valid == 0:
                continue

            # a) Pick out only the valid entries
            valid_inds = mask_indices[valid_flags]  # [num_valid, ...]
            valid_coeffs = single_coeffs[valid_flags]  # [num_valid, n_coeffs]
            pred_logits = single_pred_logits[valid_flags]  # [num_valid, H, W]
            gt_boxes = single_boxes[valid_flags]  # [num_valid, 4]

            # b) Gather the matching downsampled GT masks
            #    mask_indices are indices into the first dim of single_down_gt
            gt_masks_sel = torch.gather(
                single_down_gt,
                0,
                valid_inds.repeat(1, single_down_gt.shape[1], single_down_gt.shape[2]),
            )

            # c) Create a binary mask from boxes to zero out outside‐box pixels
            box_mask = get_tensor_mask_from_boxes(pred_logits, gt_boxes)

            # d) BCE per pixel, then apply the box mask
            pixel_bce = (
                F.binary_cross_entropy_with_logits(
                    pred_logits, gt_masks_sel, reduction="none"
                )
                * box_mask
            )

            # e) Sum & normalize by mask area
            sum_loss = pixel_bce.view(num_valid, -1).sum(dim=1)
            mask_areas = gt_masks_sel.view(num_valid, -1).sum(dim=1).float()

            normalized_loss = sum_loss / (mask_areas + 1.0)

            # f) Record into the global tensor
            per_mask_losses[loss_cursor : loss_cursor + num_valid] = normalized_loss
            loss_cursor += num_valid

            # g) Accumulate coefficient diversity loss
            diversity_accum += self.coeff_diversity_loss(valid_coeffs, valid_inds[:, 0])

        # 7) Final aggregation
        final_mask_loss = (per_mask_losses * box_norm).sum() / (classification_norm)
        final_diversity_loss = diversity_accum / all_coeffs.size(0)

        return final_mask_loss, final_diversity_loss


class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]
        self.seg_rate = loss_cfg.objective.get("LincombMaskLoss", None)
        self.coeffs_diversity_rate = loss_cfg.objective.get("CoeffsDiversityLoss", 0)

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
        aux_iou, aux_dfl, aux_cls, aux_seg, aux_coeffs_diversity = self.loss(
            aux_predicts, targets, aux_seg_logits, target_seg
        )
        main_iou, main_dfl, main_cls, main_seg, main_coeffs_diversity = self.loss(
            main_predicts, targets, main_seg_logits, target_seg
        )

        self.seg_rate = self.seg_rate or torch.tensor(0.0)

        total_loss = [
            self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            self.cls_rate * (aux_cls * self.aux_rate + main_cls),
            self.seg_rate * (aux_seg * self.aux_rate + main_seg),
            self.coeffs_diversity_rate
            * (aux_coeffs_diversity * self.aux_rate + main_coeffs_diversity),
        ]
        loss_dict = {
            f"Loss/{name}Loss": value.detach().item()
            for name, value in zip(
                ["Box", "DFL", "BCE", "LincombMask", "CoeffsDiversityLoss"], total_loss
            )
        }
        return sum(total_loss), loss_dict


def create_loss_function(cfg: Config, vec2box) -> DualLoss:
    # TODO: make it flexible, if cfg doesn't contain aux, only use SingleLoss
    loss_function = DualLoss(cfg, vec2box)
    logger.info(":white_check_mark: Success load loss function")
    return loss_function
