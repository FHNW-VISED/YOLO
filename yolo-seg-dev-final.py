# %%
%load_ext autoreload
%autoreload 2

import gc
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize
from tqdm import tqdm

# Set project root so modules are found correctly
project_root = Path().resolve().parent
sys.path.append(str(project_root))

from yolo import Config, PostProcess, create_converter, create_model
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.model_utils import get_device, get_mask_preds

from ifc_dl.conf.augmentations import get_transform_fn

# from ifc_dl.data.mock_coco_dataset import MockCocoDataModule
from ifc_dl.data.anthony_dataset import AnthonyDataModule


# %%
BASE_PATH = Path("/Users/simone.bonato/Desktop/ecolution/ecolution-floorplan-seg/")
# BASE_PATH = Path("/cluster/group/vised/ecolution/ifc_dl/code/Simone/IFC_DL/")

IMAGE_SIZE = (384, 384)
AUGS = {
    "resize": {
        "params": {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1], "interpolation": 3},
        "all_datasets": True,
    },
    # "horizontal_flip": {"params": {"p": 0.5}},
}
EPOCHS = 200
BATCH_SIZE = 2

CONFIG_PATH = "../YOLO/yolo/config"
CONFIG_NAME = "config-seg"
CLASS_NUM = 2
MODEL_WEIGHTS = BASE_PATH / "submodules/YOLO/weights/v9-c.pt"

SAVE_MODEL_PATH = BASE_PATH / "submodules/YOLO/weights/new_weights.pt"
MASK_LOSS_IMG_PATH = BASE_PATH / "submodules/YOLO/weights/mask_loss.png"


device = "cpu"


torch.cuda.empty_cache()
gc.collect()

torch.manual_seed(0)
torch.cuda.empty_cache()

# Set project root so modules are found correctly
project_root = Path().resolve().parent
sys.path.append(str(project_root))

# %%
transforms, val_transform = get_transform_fn(AUGS)
datamodule = AnthonyDataModule(
    BASE_PATH / "data/anthony",
    batch_size=BATCH_SIZE,
    transforms=transforms,
    val_transforms=val_transform,
)
datamodule.setup()
# train_dl = datamodule.train_dataloader()
train_dl = datamodule.val_dataloader()


def convert_y_for_yolo(y):
    """
    Convert a list of target dictionaries to the YOLO-specific format.

    Args:
        y (list): A batch of target dictionaries containing "labels", "boxes", and "masks".

    Returns:
        y_yolo (Tensor): Tensor with shape (batch_size, max_annotations, 5)
                         where boxes are formatted for YOLO.
        y_masks (Tensor): Tensor with the down-binary masks.
    """
    batch_size = len(y)
    max_annotations = max(len(sample["labels"]) for sample in y)
    mask_shape = y[0]["masks"].shape[-2:]
    y_yolo = torch.ones((batch_size, max_annotations, 5)) * -1
    y_masks = torch.ones((batch_size, max_annotations, *mask_shape)) * -1

    for i, sample in enumerate(y):
        for j, (label, box, mask) in enumerate(
            zip(sample["labels"], sample["boxes"], sample["masks"])
        ):
            y_yolo[i, j, 0] = label
            y_yolo[i, j, 1:] = box
            y_masks[i, j] = (mask > 0).float()
    return y_yolo, y_masks


with initialize(config_path=CONFIG_PATH, version_base=None, job_name="notebook_job"):
    cfg: Config = compose(config_name=CONFIG_NAME)

# for k in cfg.task.loss.objective:
#     cfg.task.loss.objective[k] = cfg.task.loss.objective[k] if k != "LincombMaskLoss" else 10

model = create_model(cfg.model, class_num=CLASS_NUM, weight_path=MODEL_WEIGHTS)
model = model.to(device)

converter = create_converter(
    cfg.model.name, model, cfg.model.anchor, IMAGE_SIZE, device
)

# Optionally set up post-processing if NMS is used
post_process = None
if cfg.task.get("nms"):
    post_process = PostProcess(converter, cfg.task.nms)

cfg.dataset.class_num = CLASS_NUM
loss_fn = create_loss_function(cfg, converter)

model.train()

optim = torch.optim.Adam(model.parameters())

# %%
mask_losses = []
grad_norms = []

model.train()

# Get a single batch
single_batch = next(iter(train_dl))
x, y = single_batch

# Fix input channels: repeat channels if a sample has only one channel
x = list(x)
for i in range(len(x)):
    if x[i].shape[0] == 1:
        x[i] = x[i].repeat(3, 1, 1)
x = torch.stack(x).to(device)

# Convert target annotations to YOLO format
y_yolo, y_masks = convert_y_for_yolo(y)
y_yolo = y_yolo.to(device)
y_masks = y_masks.to(device)

for epoch in range(EPOCHS):
    # Forward pass
    out = model(x)
    
    det_logits, seg_logits = out["Main"]
    det_logits_aux, seg_logits_aux = out["AUX"]

    det_preds = converter(det_logits)
    det_preds_aux = converter(det_logits_aux)

    # Compute loss
    loss_value, loss_dict = loss_fn(
        det_preds_aux,
        det_preds,
        deepcopy(y_yolo),
        y_masks,
        seg_logits_aux,
        seg_logits,
    )

    # Backpropagation and optimization step
    optim.zero_grad()
    loss_value.backward()

    # Monitor gradient norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    grad_norms.append(total_norm)

    optim.step()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss_value.item()}, Grad Norm: {total_norm}"
    )
    mask_losses.append(loss_dict["Loss/LincombMaskLoss"])

# Save model and plot
torch.save(model.state_dict(), SAVE_MODEL_PATH)
plt.figure()
plt.plot(mask_losses, label="Mask Loss")
# plt.plot(grad_norms, label="Grad Norm")
plt.legend()
plt.title("Mask Loss and Gradient Norm Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.show()

# %%
# mask_losses = []
# for epoch in range(EPOCHS):
#     tqdm_loop = tqdm(enumerate(train_dl), total=len(train_dl), desc="Training")
#     mask_losses_epoch = []
#     for batch_idx, (x, y) in tqdm_loop:
#         # Fix input channels: repeat channels if a sample has only one channel
#         x = list(x)
#         for i in range(len(x)):
#             if x[i].shape[0] == 1:
#                 x[i] = x[i].repeat(3, 1, 1)
#         x = torch.stack(x).to(device)

#         # Convert target annotations to YOLO format
#         y_yolo, y_masks = convert_y_for_yolo(y)
#         y_yolo = y_yolo.to(device)
#         y_masks = y_masks.to(device)

#         # Forward pass
#         out = model(x)
#         det_logits, seg_logits = out["Main"]
#         det_logits_aux, seg_logits_aux = out["AUX"]

#         det_preds = converter(det_logits)
#         det_preds_aux = converter(det_logits_aux)

#         # Compute loss
#         loss_value, loss_dict = loss_fn(
#             det_preds_aux,
#             det_preds,
#             deepcopy(y_yolo),
#             y_masks,
#             seg_logits_aux,
#             seg_logits,
#         )
#         tqdm_loop.set_description(
#             f"Epoch: {epoch + 1} | Batch {batch_idx + 1}/{len(train_dl)} | {loss_dict=}"
#         )

#         # Backpropagation and optimization step
#         optim.zero_grad()
#         loss_value.backward()
#         optim.step()

#         mask_losses_epoch.append(loss_dict["Loss/LincombMaskLoss"])

#         if batch_idx == 3:
#             break

#     mask_losses.append(torch.mean(torch.tensor(mask_losses_epoch)).item())

# %%
pred = model(x)
det_logits, seg_logits = pred["Main"]
det_logits_aux, seg_logits_aux = pred["AUX"]

det_preds = converter(det_logits)
det_preds_aux = converter(det_logits_aux)

# Get the mask predictions
mask_preds = get_mask_preds(seg_logits, True)


# %%
from omegaconf import OmegaConf

nms_config = {
    "min_confidence": 0.03,
    "min_iou": 0.3,
    "max_bbox": 300,
}
nms_config = OmegaConf.create(nms_config)
post_proccess = PostProcess(converter, nms_config)

boxes, seg = post_proccess(pred, seg_threshold=0.5)

# %%
img_idx = 1

plt.subplot(1, 2, 1)
plt.imshow(y_masks[img_idx].sum(0).cpu().numpy())

plt.subplot(1, 2, 2)
plt.imshow(seg[img_idx].sum(0).cpu().numpy() > 0)
plt.show()



# %%
