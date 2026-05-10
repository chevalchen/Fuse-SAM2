r""" GeoCrack few-shot semantic segmentation dataset """
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetGeoCrack(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=600):
        self.split = "test" if split in ["val", "test"] else "trn"
        self.benchmark = "geocrack"
        self.fold = fold
        self.shot = shot
        self.transform = transform
        self.num = num
        self.nclass = 1
        self.nfolds = 4

        self.base_path = Path(datapath) / "geocrack"
        self.patch_dir = self.base_path / "patched_images"
        self.patch_pairs_file = self.base_path / "patch_pairs.tab"

        self.categories = ["1"]
        self.class_ids = range(0, 1)

        self.samples = self.build_samples()
        if len(self.samples) <= self.shot:
            raise ValueError(
                f"GeoCrack split '{self.split}' only has {len(self.samples)} samples, "
                f"which is not enough for {self.shot}-shot episodes."
            )

        self.img_metadata_classwise = {"1": [sample["image_path"] for sample in self.samples]}
        self.mask_map = {sample["image_path"]: sample["mask_path"] for sample in self.samples}
        print(f"Total ({self.split}) {self.benchmark} images are : {len(self.img_metadata_classwise['1'])}")

    def __len__(self):
        if self.split == "trn":
            return self.num
        return len(self.img_metadata_classwise["1"])

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(
            query_mask.unsqueeze(0).unsqueeze(0).float(),
            query_img.size()[-2:],
            mode="nearest",
        ).squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(
                smask.unsqueeze(0).unsqueeze(0).float(),
                support_imgs.size()[-2:],
                mode="nearest",
            ).squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {
            "query_img": query_img,
            "query_mask": query_mask,
            "query_name": query_name,
            "support_imgs": support_imgs,
            "support_masks": support_masks,
            "support_names": support_names,
            "class_id": torch.tensor(class_sample),
        }
        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert("RGB")
        support_imgs = [Image.open(name).convert("RGB") for name in support_names]
        query_mask = self.read_mask(self.mask_map[query_name])
        support_masks = [self.read_mask(self.mask_map[name]) for name in support_names]
        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert("L")))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        candidates = self.img_metadata_classwise[self.categories[class_id]]

        if self.split == "trn":
            query_name = np.random.choice(candidates, 1, replace=False)[0]
        else:
            query_name = candidates[idx % len(candidates)]

        support_names = []
        while len(support_names) < self.shot:
            support_name = np.random.choice(candidates, 1, replace=False)[0]
            if support_name == query_name or support_name in support_names:
                continue
            support_names.append(support_name)

        return query_name, support_names, class_id

    def build_samples(self):
        paired_samples = self._load_patch_pairs()
        grouped = defaultdict(list)
        for sample in paired_samples:
            grouped[sample["stem"]].append(sample)

        stems = sorted(grouped.keys())
        if not stems:
            raise FileNotFoundError(f"No valid GeoCrack patch pairs found under {self.patch_dir}")

        if self.split == "trn":
            selected_stems = self._get_train_stems(stems)
        else:
            selected_stems = self._get_eval_stems(stems)

        samples = []
        for stem in selected_stems:
            samples.extend(sorted(grouped[stem], key=lambda item: item["image_path"]))
        return samples

    def _get_train_stems(self, stems):
        if self.fold == -1:
            return stems
        eval_stems = set(self._get_eval_stems(stems))
        return [stem for stem in stems if stem not in eval_stems]

    def _get_eval_stems(self, stems):
        if self.fold == -1:
            return stems
        fold = self.fold % self.nfolds
        return stems[fold::self.nfolds]

    def _load_patch_pairs(self):
        if self.patch_pairs_file.exists():
            samples = self._load_patch_pairs_from_file()
            if samples:
                return samples

        return self._load_patch_pairs_from_glob()

    def _load_patch_pairs_from_file(self):
        samples = []
        with open(self.patch_pairs_file, "r", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                image_name = Path(row[0].strip().strip('"').replace("\\", "/")).name
                mask_name = Path(row[1].strip().strip('"').replace("\\", "/")).name
                image_path = self.patch_dir / image_name
                mask_path = self.patch_dir / mask_name
                if not image_path.exists() or not mask_path.exists():
                    continue
                samples.append(self._build_sample(image_path, mask_path))
        return samples

    def _load_patch_pairs_from_glob(self):
        samples = []
        for image_path in sorted(self.patch_dir.glob("*_original_patch*.png")):
            mask_name = image_path.name.replace("_original_patch", "_binarymask_patch")
            mask_path = image_path.with_name(mask_name)
            if mask_path.exists():
                samples.append(self._build_sample(image_path, mask_path))
        return samples

    def _build_sample(self, image_path, mask_path):
        image_path = str(image_path)
        mask_path = str(mask_path)
        stem = os.path.basename(image_path).split("_original_patch")[0]
        return {
            "image_path": image_path,
            "mask_path": mask_path,
            "stem": stem,
        }


def build(image_set, args):
    img_size = 518
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = DatasetGeoCrack(
        datapath=args.data_root,
        fold=args.fold,
        transform=transform,
        shot=args.shots,
        split=image_set,
    )
    return dataset
