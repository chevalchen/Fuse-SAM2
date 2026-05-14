from typing import Dict

import torch
import torch.nn.functional as F
import einops


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_boxes: int) -> torch.Tensor:
    """
    Compute the Dice loss for binary masks.

    Args:
        inputs (Tensor): raw logits, shape [N, ...].
        targets (Tensor): binary masks with same shape as inputs.
        num_boxes (int): normalization factor (usually batch size).

    Returns:
        Tensor: scalar Dice loss.
    """
    inputs = inputs.sigmoid().flatten(1).float()
    targets = targets.flatten(1).float()

    numerator   = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Compute the sigmoid focal loss (used in RetinaNet).

    Args:
        inputs (Tensor): raw logits, shape [N, ...].
        targets (Tensor): binary masks with same shape as inputs.
        num_boxes (int): normalization factor (usually batch size).
        alpha (float): class balancing factor (0–1). Default: 0.25.
        gamma (float): focusing parameter. Default: 2.0.

    Returns:
        Tensor: scalar focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def loss_masks(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    num_frames: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute focal and dice loss on predicted masks.

    Args:
        outputs (Tensor): B*T, h, W.
        masks (Tensor): B, T, H, W (binary 0/1).
        num_frames (int): number of last frames to include (T → all, T-1 → skip first).

    Returns:
        dict: {"loss_mask": Tensor, "loss_dice": Tensor}
    """
    bs, T = masks.shape[:2]
    start = max(0, T - num_frames)

    src = einops.rearrange(outputs, '(b t) h w -> b t h w', b=bs)[:, start:T]
    tgt = masks[:, start:]

    # flatten to [B, F*H*W]
    tgt = tgt.to(src)
    src = src.flatten(1).to(torch.float32)
    tgt = tgt.flatten(1).to(src.dtype)

    # drop NaN rows (if any)
    keep = ~torch.isnan(src).any(dim=1)
    if not keep.all():
        src, tgt = src[keep], tgt[keep]
        bs = int(keep.sum())

    return {
        "loss_mask": sigmoid_focal_loss(src, tgt, bs),
        "loss_dice":  dice_loss(src, tgt, bs),
    }


def uncertainty_nll_loss(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    prob = torch.sigmoid(pred_logits)
    gt = gt_masks.to(device=pred_logits.device, dtype=pred_logits.dtype)

    if log_var.dim() == 4:
        log_var = log_var.squeeze(1)
    elif log_var.dim() != 3:
        raise ValueError(f"Unsupported log_var shape: {tuple(log_var.shape)}")

    log_var = log_var.clamp(-6.0, 6.0)
    err_sq = (prob - gt).pow(2)
    loss = 0.5 * torch.exp(-log_var) * err_sq + 0.5 * log_var
    return loss.mean()


def uncertainty_loss(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    log_var: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    bs, t = gt_masks.shape[:2]
    start = max(0, t - num_frames)

    pred = pred_masks[:, start:t].reshape(-1, *pred_masks.shape[-2:])
    gt = gt_masks[:, start:t].reshape(-1, *gt_masks.shape[-2:])
    if log_var.dim() == 5:
        log_var = log_var[:, start:t].reshape(-1, *log_var.shape[-3:])

    return uncertainty_nll_loss(pred_logits=pred, gt_masks=gt, log_var=log_var)
