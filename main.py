import argparse
import copy
import json
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util.commons import resume_from_checkpoint, adapter_state_dict
import datasets.samplers as samplers
from datasets import build_dataset
from models.sansa.sansa import build_sansa
from inference_fss import eval_fss
from engine import train_one_epoch
import opts
from util.commons import setup_logging, make_deterministic


def main(args):
    utils.init_distributed_mode(args)
    rank = utils.get_rank()
    setup_logging(save_dir=args.output_dir, console="info", rank=rank)
    make_deterministic(args.seed+rank) # fix the seed for reproducibility

    print(args)

    device = torch.device(args.device)
    model = build_sansa(
        args.sam2_version,
        args.adaptformer_stages,
        args.channel_factor,
        args.device,
        use_uncertainty=args.use_uncertainty,
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters_tot = sum(p.numel() for p in model.parameters())
    print(f'number of params: {n_parameters_tot}')

    # --- Parameter groups ---
    # In fine-tune mode with differential LR, peek at the checkpoint to learn which
    # adapter params are warm-started (in checkpoint) vs newly added (e.g. Stage-1
    # adapters that were not in the original generalist.pth).  Warm-started params
    # get lr * finetune_lr_scale; new params get the full lr.
    _ck_keys: set = set()
    if args.finetune and args.resume and abs(args.finetune_lr_scale - 1.0) > 1e-6:
        _raw_ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        _ck_keys = set(_raw_ck.get('model', {}).keys())
        del _raw_ck  # free before training starts

    head_warmstart, head_new, fix = [], [], []
    for name, p in model_without_ddp.named_parameters():
        if p.requires_grad:
            (head_warmstart if name in _ck_keys else head_new).append(p)
        else:
            fix.append(p)

    n_train = sum(p.numel() for p in head_warmstart) + sum(p.numel() for p in head_new)
    print(f'Trainable parameters: {n_train}')
    print(f'Parameters fixed: {sum(p.numel() for p in fix)}')

    if head_warmstart:
        lr_ws = args.lr * args.finetune_lr_scale
        param_list = [
            {'params': head_warmstart, 'initial_lr': lr_ws},
            {'params': head_new,       'initial_lr': args.lr},
        ]
        print(f'Differential LR: {len(head_warmstart)} warm-start params @ {lr_ws:.2e}, '
              f'{len(head_new)} new params @ {args.lr:.2e}')
    else:
        param_list = [{'params': head_new, 'initial_lr': args.lr}]

    optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), fused=True)

    cfg = copy.deepcopy(args)
    cfg.shots = args.J
    dataset_train = build_dataset(args.dataset_file, image_set='train', args=cfg)

    args.batch_size = int(args.batch_size / args.ngpu)
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(data_loader_train)
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.finetune:
            # Load model weights only; optimizer/LR state from old checkpoint is
            # incompatible when adapter stages or param counts differ.
            resume_from_checkpoint(args.resume, model_without_ddp)
        else:
            model_without_ddp, optimizer, lr_scheduler = resume_from_checkpoint(
                args.resume, model_without_ddp, optimizer, lr_scheduler, args)
        
    print("Start training")
    best_miou = -1.0
    best_epoch = -1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if args.use_uncertainty:
            unc_recalib_active = epoch >= args.uncertainty_warmup_epochs
            if hasattr(model, "module"):
                model.module.unc_recalib_active = unc_recalib_active
            else:
                model.unc_recalib_active = unc_recalib_active

        train_stats = train_one_epoch(
                model, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, lr_scheduler=lr_scheduler, args=args)

        print(f"Start validation")
        miou = eval_fss(model, args)
        is_best = miou > best_miou
        if is_best and args.output_dir:
            best_miou = miou
            best_epoch = epoch
            checkpoint_path = output_dir / 'checkpoint_best.pth'
            utils.save_on_master({
                'model': adapter_state_dict(model_without_ddp),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_miou': best_miou,
                'args': args,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters_tot,
                     'val_miou': miou,
                     'best_miou': best_miou,
                     'best_epoch': best_epoch}

        print(json.dumps(log_stats))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA training', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.name_exp)

    main(args)
