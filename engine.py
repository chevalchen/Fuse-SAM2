import math
import random
import sys
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer

import util.misc as utils
from util.losses import loss_masks, uncertainty_nll_loss
from util.promptable_utils import build_prompt_dict


def train_one_epoch(
    model: Module,
    data_loader: Iterable,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    args: Optional[object] = None,
) -> dict:
    """
    Run one training epoch.

    Args:
        model (nn.Module): the model to train.
        data_loader (Iterable): yields training batches.
        optimizer (Optimizer): optimizer for parameter updates.
        device (torch.device): device to run computations on.
        epoch (int): current epoch index.
        max_norm (float): gradient clipping norm (0 = no clipping).
        lr_scheduler (optional): learning rate scheduler to step each iteration.
        args (Namespace): experiment/training arguments.
    Returns:
        dict: training statistics.
    """

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        model.train()
        # unpack
        query_img = batch['query_img']         # [B, C, H, W]
        query_mask = batch['query_mask']       # [B, H, W]
        support_imgs = batch['support_imgs']   # [B, S, C, H, W]
        support_masks = batch['support_masks'] # [B, S, H, W]

        random_shot = random.randint(1, args.shots)

        # build sequence: supports + query  -> T = random_shot + 1
        samples = torch.cat([support_imgs[:, :random_shot], query_img.unsqueeze(1)], dim=1)   # [B, T, C, H, W]
        masks   = torch.cat([support_masks[:, :random_shot], query_mask.unsqueeze(1)], dim=1) # [B, T, H, W]

        prompt_dict = build_prompt_dict(masks, args.prompt, n_shots=random_shot, train_mode=True, device=model.device)
        samples = samples.to(device)

        outputs = model(samples, prompt_dict)

        # loss on last T or last (T-1) frames (skip first) depending on prompt
        T = samples.shape[1]
        use_frames = T if args.prompt != "mask" else (T - 1)
        losses_dict = loss_masks(outputs["pred_masks"], masks, num_frames=use_frames)
        if "uncertainty_feat" in outputs and outputs["uncertainty_feat"] is not None:
            bs = masks.shape[0]
            pred_bt = outputs["pred_masks"].view(bs, T, *outputs["pred_masks"].shape[-2:])
            gt_bt = masks.to(pred_bt.device)
            q_frames = max(1, outputs["uncertainty_feat"].shape[0] // bs)
            pred_query = pred_bt[:, -q_frames:].reshape(-1, *pred_bt.shape[-2:])
            gt_query = gt_bt[:, -q_frames:].reshape(-1, *gt_bt.shape[-2:])
            unc_feat = outputs["uncertainty_feat"]

            target_size = unc_feat.shape[-2:]
            pred_small = F.interpolate(
                pred_query.unsqueeze(1),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            gt_small = F.interpolate(
                gt_query.unsqueeze(1).float(),
                size=target_size,
                mode="nearest",
            ).squeeze(1)

            unc_loss = uncertainty_nll_loss(
                pred_logits=pred_small,
                gt_masks=gt_small,
                log_var=unc_feat,
            )
            losses_dict["loss_uncertainty"] = unc_loss * args.uncertainty_loss_weight

        # total loss
        loss = sum(losses_dict.values())

        # reduce for logging
        losses_reduced = utils.reduce_dict(losses_dict)
        loss_value = sum(losses_reduced.values()).item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(losses_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True) # avoids unnecessary memory allocations for zero grad tensor
        loss.backward()

        grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, **losses_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
