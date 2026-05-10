import argparse
import sys
from pathlib import Path
from os.path import join
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import opts
from models.sansa.sansa import build_sansa
from datasets import build_dataset
from util.commons import make_deterministic, setup_logging, resume_from_checkpoint
import util.misc as utils
from util.promptable_utils import build_prompt_dict
from util.metrics import AverageMeter, Evaluator
from tqdm import tqdm
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam_fuse.parent_finder import (
    compute_granularity_image_relative,
    compute_granularity_sam2,
)


def _assert_vanilla_inference_state(model: torch.nn.Module, args: argparse.Namespace) -> None:
    vanilla_mode = (not args.use_granularity) or args.granularity_source == "none"
    if not vanilla_mode or not hasattr(model.sam, "gra_mem_scale"):
        return

    scale = float(model.sam.gra_mem_scale.detach().cpu().item())
    if abs(scale) > 1e-12:
        raise RuntimeError(
            "Vanilla evaluation requires gra_mem_scale == 0.0, but the loaded checkpoint "
            f"has gra_mem_scale={scale:.6f}. Re-run with a clean vanilla checkpoint or "
            "reset the granularity weights before evaluation."
        )


def _resolve_inference_granularities(model: torch.nn.Module, batch, args):
    if not args.use_granularity:
        return None

    if args.granularity_source == "none":
        return None

    if args.granularity_source == "gt" and "granularity" in batch:
        return batch["granularity"]

    support_imgs = batch["support_imgs"][0]
    support_masks = batch["support_masks"][0]

    values = []
    for support_img, support_mask in zip(support_imgs, support_masks):
        if args.granularity_source == "sam2" and hasattr(model.sam, "automatic_mask_generator"):
            try:
                values.append(
                    compute_granularity_sam2(
                        reference_mask=support_mask,
                        reference_image=support_img,
                        sam2_model=model.sam.automatic_mask_generator,
                    )
                )
                continue
            except Exception:
                pass
        values.append(compute_granularity_image_relative(support_mask, support_img))

    return torch.tensor([values], dtype=torch.float32)


def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    print(args)

    model = build_sansa(
        args.sam2_version,
        args.adaptformer_stages,
        args.channel_factor,
        args.device,
        use_granularity=args.use_granularity,
        sansa_checkpoint=args.sansa_checkpoint,
        load_granularity_from_sansa_checkpoint=args.load_granularity_from_sansa_checkpoint,
        granularity_mlp_init=args.granularity_mlp_init,
        unsamv2_ckpt_path=args.unsamv2_ckpt_path,
    )
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)

    _assert_vanilla_inference_state(model, args)

    print(f"number of params: {n_parameters}")
    print('Start inference')

    mIoU = eval_fss(model, args)
    return mIoU


def eval_fss(model: torch.nn.Module, args: argparse.Namespace) -> float:
    """
    Evaluate SANSA on the few-shot segmentation benchmark.
    Computes and prints mIoU across the validation set.
    """
    # load data
    validation_ds = 'coco' if args.dataset_file == 'multi' else args.dataset_file 
    print(f'Evaluating {validation_ds} - fold: {args.fold}')
    ds = build_dataset(validation_ds, image_set='val', args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    average_meter = AverageMeter(args.dataset_file, ds.class_ids, ds.nclass)

    pbar = tqdm(dataloader, ncols=80, desc='runn avg.', disable=(utils.get_rank() != 0), file=sys.stderr, dynamic_ncols=True)
    for idx, batch in enumerate(pbar):
        query_img, query_mask = batch['query_img'], batch['query_mask']
        support_imgs, support_masks = batch['support_imgs'], batch['support_masks']
        granularities = _resolve_inference_granularities(model, batch, args)

        imgs = torch.cat([support_imgs[0], query_img]).unsqueeze(0) # b t c h w
        img_h, img_w = imgs.shape[-2:]

        imgs = imgs.to(args.device)
        prompt_dict = build_prompt_dict(support_masks, args.prompt, n_shots=args.shots, train_mode=False, device=model.device)

        with torch.no_grad():
            outputs = model(imgs, prompt_dict, granularities=granularities)

        pred_masks = outputs["pred_masks"].unsqueeze(0)  # [1, T, h, w]
        pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode='bilinear', align_corners=False) 
        pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu()

        area_inter, area_union = Evaluator.classify_prediction(pred_masks[-1:].float(), batch, device=imgs.device)
        average_meter.update(area_inter, area_union, batch['class_id'].cuda())

        if (idx + 1) % 50 == 0:
            miou, _, _ = average_meter.compute_iou()
            pbar.set_description(f"Runn. Avg mIoU = {miou:.1f}")

        if args.visualize:
            from util.visualization import visualize_episode
            fg_inter = area_inter[1].sum().item()
            fg_union = area_union[1].sum().item()
            episode_iou = fg_inter / max(fg_union, 1e-6)
            visualize_episode(
                support_imgs=[support_imgs[0, i].cpu() for i in range(args.shots)],
                query_img=query_img[0].cpu(),
                query_gt=(query_mask[0].numpy() > 0),
                query_pred=pred_masks[-1].numpy(),
                prompt_dict=prompt_dict,
                out_dir=args.output_dir,
                idx=idx,
                src_size=model.sam.image_size,
                iou=episode_iou,
            )
    average_meter.write_result(args.dataset_file)
    miou, fb_iou, _ = average_meter.compute_iou()
    print('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, miou, fb_iou.item()))
    print('==================== Finished Testing ====================')

    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)
