import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.loss_functions import DualLoss, create_loss_function
from yolo.utils.bounding_box_utils import Vec2Box


@pytest.fixture
def cfg() -> Config:
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg = compose(config_name="config", overrides=["task=train"])
    return cfg


@pytest.fixture
def model(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model, weight_path=None)
    return model.to(device)


@pytest.fixture
def vec2box(cfg: Config, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Vec2Box(model, cfg.model.anchor, cfg.image_size, device)


@pytest.fixture
def loss_function(cfg, vec2box) -> DualLoss:
    return create_loss_function(cfg, vec2box)


@pytest.fixture
def data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = torch.zeros(1, 20, 5, device=device)
    predicts = [
        torch.zeros(1, 8400, *cn, device=device) for cn in [(80,), (4, 16), (4,)]
    ]
    return predicts, targets


def test_yolo_loss(loss_function, data):
    predicts, targets = data
    loss, loss_dict = loss_function(predicts, predicts, targets)
    assert loss_dict["Loss/BoxLoss"] == 0
    assert loss_dict["Loss/DFLLoss"] == 0
    assert loss_dict["Loss/BCELoss"] >= 2e5


import pytest
import torch

# Import the MaskLoss class from your module.
# Adjust the import path as needed.
from yolo.tools.loss_functions import MaskLoss

# Define dummy implementations for the external helper functions.
# These are used to bypass the real implementations so that
# the forward pass can be tested in isolation.


def dummy_get_mask_preds(pred_logits, sigmoid=False):
    """
    Dummy replacement for get_mask_preds.
    For simplicity, return the sigmoid of the last element
    in the pred_logits list.
    """
    if sigmoid:
        # pred_logits[-1] has shape [batch, anchors, h, w]
        return torch.sigmoid(pred_logits[-1])
    return pred_logits[-1]


def dummy_reshape_batched_bboxes(gt_boxes, orig_size, new_size):
    """
    Dummy replacement for reshape_batched_bboxes.
    Simply unsqueeze along a new dimension so that later indexing works.
    """
    # Here, gt_boxes is assumed to be of shape (batch, num_preds, 4)
    return gt_boxes.unsqueeze(1)


def dummy_mask_tensor_with_boxes(valid_pred, matched_gt_boxes):
    """
    Dummy replacement for mask_tensor_with_boxes.
    Simply return the predictions unchanged.
    """
    return valid_pred


# Apply monkeypatch to override the external functions used by MaskLoss.
@pytest.fixture(autouse=True)
def patch_external_functions(monkeypatch):
    # Since MaskLoss imports these functions at the module level,
    # we patch them in its module namespace.
    import yolo.tools.loss_functions as lf

    monkeypatch.setattr(lf, "get_mask_preds", dummy_get_mask_preds)
    monkeypatch.setattr(lf, "reshape_batched_bboxes", dummy_reshape_batched_bboxes)
    monkeypatch.setattr(lf, "mask_tensor_with_boxes", dummy_mask_tensor_with_boxes)


def test_mask_loss_normal():
    """
    Test the forward pass of MaskLoss in a scenario with one valid prediction.

    For this test, we set:
      - pred_logits to zeros so that sigmoid(0)==0.5.
      - targets to ones so that after interpolation and binarization they are 1s.
      - valid_masks with one True value.
      - gt_boxes with a dummy bounding box.

    Therefore, since BCE loss with input 0.5 against target 1 produces -log(0.5) ≃ 0.6931 per pixel,
    after summing over the pixels and normalizing by the area (which cancels out), we expect the
    final loss to be approximately 0.6931.
    """
    batch_size = 1
    num_preds = 1  # maximum predictions per batch sample (for targets)
    total_anchors = 1  # one anchor per sample
    proto_h, proto_w = 64, 64
    orig_h, orig_w = 128, 128

    # Create a list of predicted logits.
    # We only need one element; its shape is (batch, anchors, proto_h, proto_w).
    # Setting logits to zero causes sigmoid(0)==0.5.
    pred_logits = [torch.zeros((batch_size, total_anchors, proto_h, proto_w))]

    # Targets tensor: shape (batch, num_preds, orig_h, orig_w).
    # All ones so that after bilinear interpolation and thresholding,
    # the downsampled masks become full of ones.
    targets = torch.ones((batch_size, num_preds, orig_h, orig_w))

    # Boolean tensor for valid masks: shape (batch, total_anchors).
    valid_masks = torch.tensor([[True]])

    # Anchor-to-ground-truth indices: shape (batch, total_anchors, 1).
    # Here, we map the only anchor to the first (and only) ground-truth object (index 0).
    anchor_to_gt_idxs = torch.zeros((batch_size, total_anchors, 1), dtype=torch.long)

    # Ground truth boxes: shape (batch, num_preds, 5).
    # For example, the first coordinate can be a dummy label (0) and the rest are box coordinates.
    gt_boxes = torch.tensor([[[0, 10.0, 20.0, 30.0, 40.0]]])

    # Normalization constants.
    cls_norm = torch.tensor(1.0)
    box_norm = torch.tensor(1.0)

    loss_fn = MaskLoss()
    loss = loss_fn(
        pred_logits,
        targets,
        valid_masks,
        anchor_to_gt_idxs,
        gt_boxes,
        cls_norm,
        box_norm,
    )
    # Expectation: For an input of 0.5 vs. target of 1, the per-pixel binary cross entropy loss is about 0.6931.
    expected_loss = 0.6931
    assert torch.isclose(
        loss, torch.tensor(expected_loss), atol=1e-3
    ), f"Expected loss approximately {expected_loss}, got {loss.item()}"


def test_mask_loss_no_valid():
    """
    Test the forward pass of MaskLoss when no anchors are valid.

    This test sets the valid_masks to all False. Since the forward method computes:
        final_loss = (total_losses * box_norm).sum() / (cls_norm * valid_masks.sum())
    the denominator becomes zero and the result should be NaN.
    """
    batch_size = 1
    num_preds = 1
    total_anchors = 1
    proto_h, proto_w = 64, 64
    orig_h, orig_w = 128, 128

    pred_logits = [torch.zeros((batch_size, total_anchors, proto_h, proto_w))]
    targets = torch.ones((batch_size, num_preds, orig_h, orig_w))
    valid_masks = torch.tensor([[False]])
    anchor_to_gt_idxs = torch.zeros((batch_size, total_anchors, 1), dtype=torch.long)
    gt_boxes = torch.tensor([[[0, 10.0, 20.0, 30.0, 40.0]]])
    cls_norm = torch.tensor(1.0)
    box_norm = torch.tensor(1.0)

    loss_fn = MaskLoss()
    loss = loss_fn(
        pred_logits,
        targets,
        valid_masks,
        anchor_to_gt_idxs,
        gt_boxes,
        cls_norm,
        box_norm,
    )
    assert torch.isnan(
        loss
    ), f"Expected loss to be NaN when there are no valid predictions, got {loss.item()}"
