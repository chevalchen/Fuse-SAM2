import argparse
import sys
from os.path import join
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from models.sansa.sansa import build_sansa
from datasets import build_dataset
from util.commons import make_deterministic, setup_logging, resume_from_checkpoint
import util.misc as utils
from util.promptable_utils import build_prompt_dict
from util.metrics import AverageMeter, Evaluator
from util.tta_utils import (
    build_tta_passes, get_permutations, permute_supports,
    drop_small_components, binarize, calibrate_threshold_by_support,
)


def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    print(args)

    model = build_sansa(
        args.sam2_version,
        args.adaptformer_stages,
        args.channel_factor,
        args.device,
        use_uncertainty=args.use_uncertainty,
    )
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)

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

        img_h, img_w = query_img.shape[-2:]

        # Build identity prompt_dict once — used for visualization.
        prompt_dict = build_prompt_dict(
            support_masks, args.prompt, n_shots=args.shots,
            train_mode=False, device=model.device,
        )

        # Module D: compute effective threshold (area-calibrated or base).
        eff_threshold = (
            calibrate_threshold_by_support(args.threshold, support_masks)
            if args.area_calibrate_threshold else args.threshold
        )

        # --- Ensemble loop: support permutations × geometric TTA passes ---
        with torch.no_grad():
            permutations = get_permutations(args.shots, args.shot_permutations)
            prob_sum = None
            n_passes = 0

            for perm in permutations:
                s_imgs, s_masks = permute_supports(support_imgs, support_masks, perm)
                imgs_perm = torch.cat([s_imgs[0], query_img]).unsqueeze(0).to(args.device)
                prompt_perm = build_prompt_dict(
                    s_masks, args.prompt, n_shots=args.shots,
                    train_mode=False, device=model.device,
                )

                for imgs_aug, inv_fn in build_tta_passes(
                    args.tta, args.tta_scales, imgs_perm, img_h, img_w
                ):
                    out = model(imgs_aug, prompt_perm)
                    logits = F.interpolate(
                        out["pred_masks"].unsqueeze(0),
                        size=(img_h, img_w), mode='bilinear', align_corners=False,
                    )
                    query_prob = inv_fn(logits.sigmoid()[0, -1])   # [H, W]
                    prob_sum = query_prob if prob_sum is None else prob_sum + query_prob
                    n_passes += 1

        prob_mean = (prob_sum / n_passes).cpu()                    # [H, W]

        # --- Post-processing (Pass 1) ---
        query_pred = binarize(prob_mean, eff_threshold, args.adaptive_threshold)
        query_pred = torch.from_numpy(
            drop_small_components(query_pred.numpy(), args.postprocess_min_area)
        ).bool()

        # --- Self-prompting refinement (Pass 2) ---
        if args.self_refine and query_pred.sum().item() >= args.self_refine_min_area:
            # Treat the Pass-1 predicted mask as a K+1-th pseudo-support.
            pseudo_mask = query_pred.float().unsqueeze(0).unsqueeze(0)          # [1, 1, H, W]
            s_imgs_r = torch.cat([support_imgs, query_img.unsqueeze(1)], dim=1) # [1, K+1, C, H, W]
            s_masks_r = torch.cat(
                [support_masks, pseudo_mask], dim=1
            )                                                                    # [1, K+1, H, W]
            imgs_r = torch.cat([s_imgs_r[0], query_img]).unsqueeze(0).to(args.device)
            prompt_r = build_prompt_dict(
                s_masks_r, 'mask', n_shots=args.shots + 1,
                train_mode=False, device=model.device,
            )
            with torch.no_grad():
                out_r = model(imgs_r, prompt_r)
                logits_r = F.interpolate(
                    out_r["pred_masks"].unsqueeze(0),
                    size=(img_h, img_w), mode='bilinear', align_corners=False,
                )
                prob_r = logits_r.sigmoid()[0, -1].cpu()                         # [H, W]
            query_pred = binarize(prob_r, eff_threshold, args.adaptive_threshold)
            query_pred = torch.from_numpy(
                drop_small_components(query_pred.numpy(), args.postprocess_min_area)
            ).bool()

        area_inter, area_union = Evaluator.classify_prediction(
            query_pred.unsqueeze(0).float(), batch, device=args.device,
        )
        average_meter.update(area_inter, area_union, batch['class_id'].cuda())

        if (idx + 1) % 50 == 0:
            miou, _, _ = average_meter.compute_iou()
            pbar.set_description(f"Runn. Avg mIoU = {miou:.1f}")

        if args.visualize:
            from util.visualization import visualize_episode
            fg_inter = area_inter[1].sum().item()
            fg_union = area_union[1].sum().item()
            vis_iou = fg_inter / max(fg_union, 1e-6)
            visualize_episode(
                support_imgs=[support_imgs[0, i].cpu() for i in range(args.shots)],
                query_img=query_img[0].cpu(),
                query_gt=(query_mask[0].numpy() > 0),
                query_pred=query_pred.numpy(),
                prompt_dict=prompt_dict,
                out_dir=args.output_dir,
                idx=idx,
                src_size=model.sam.image_size,
                iou=vis_iou,
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
