"""
Inference-time augmentation and post-processing utilities for part-seg FSS.

Three modules:
  A. Geometric TTA: flip + scale (center-crop zoom-in / zero-pad zoom-out)
  B. Support permutation ensembling (K-shot memory-bank order diversity)
  C. Post-processing: connected-component filter + Otsu adaptive threshold
"""
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label


# ---------------------------------------------------------------------------
# Module A — Geometric TTA
# ---------------------------------------------------------------------------

def build_tta_passes(
    tta_list: List[str],
    tta_scales: List[float],
    imgs: torch.Tensor,
    img_h: int,
    img_w: int,
) -> List[Tuple[torch.Tensor, Callable]]:
    """
    Return a list of (augmented_imgs, inverse_fn) pairs for all active TTA
    combinations.

    inverse_fn: maps a query prob map [H, W] in augmented space back to the
    original [H, W] coordinate frame so that passes can be averaged.

    tta_list: subset of {'none', 'flip', 'scale'}
    tta_scales: extra scale factors used when 'scale' is active (1.0 is always
        included implicitly; duplicates of 1.0 in this list are ignored)
    """
    do_flip = 'flip' in tta_list
    do_scale = 'scale' in tta_list

    scales = [1.0]
    if do_scale:
        for s in tta_scales:
            if abs(s - 1.0) > 1e-6:
                scales.append(s)

    passes = []
    for scale in scales:
        imgs_s, inv_s = _apply_scale(imgs, scale, img_h, img_w)
        # identity flip
        passes.append((imgs_s, inv_s))
        # horizontal flip (query frame only)
        if do_flip:
            imgs_sf = imgs_s.clone()
            imgs_sf[0, -1] = imgs_sf[0, -1].flip(-1)
            def _make_inv_flip(base_inv):
                def _inv(prob):
                    return base_inv(prob.flip(-1))
                return _inv
            passes.append((imgs_sf, _make_inv_flip(inv_s)))

    return passes


def _apply_scale(
    imgs: torch.Tensor,
    scale: float,
    img_h: int,
    img_w: int,
) -> Tuple[torch.Tensor, Callable]:
    """
    Apply a zoom augmentation to the query frame (last frame) only.

    scale > 1 — zoom in: center-crop to (H/s, W/s), resize back to (H, W).
        The part appears larger within SAM2's canvas.
    scale < 1 — zoom out: shrink to (H*s, W*s), zero-pad back to (H, W).
        Adds surrounding context.
    scale = 1 — identity; returned inverse is also identity.

    Returns (modified_imgs_clone, inverse_fn).
    """
    if abs(scale - 1.0) < 1e-6:
        return imgs, lambda p: p

    imgs_aug = imgs.clone()
    q = imgs_aug[0, -1]  # [C, H, W]

    if scale > 1.0:
        crop_h = int(round(img_h / scale))
        crop_w = int(round(img_w / scale))
        top = (img_h - crop_h) // 2
        left = (img_w - crop_w) // 2
        cropped = q[:, top:top + crop_h, left:left + crop_w]
        imgs_aug[0, -1] = F.interpolate(
            cropped.unsqueeze(0), size=(img_h, img_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)

        def inv_zoom_in(prob, _t=top, _l=left, _ch=crop_h, _cw=crop_w):
            small = F.interpolate(
                prob[None, None].float(), size=(_ch, _cw),
                mode='bilinear', align_corners=False,
            )[0, 0]
            out = torch.zeros(img_h, img_w, device=prob.device, dtype=small.dtype)
            out[_t:_t + _ch, _l:_l + _cw] = small
            return out

        return imgs_aug, inv_zoom_in

    else:  # scale < 1
        small_h = int(round(img_h * scale))
        small_w = int(round(img_w * scale))
        top = (img_h - small_h) // 2
        left = (img_w - small_w) // 2
        shrunk = F.interpolate(
            q.unsqueeze(0), size=(small_h, small_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)
        padded = torch.zeros_like(q)
        padded[:, top:top + small_h, left:left + small_w] = shrunk
        imgs_aug[0, -1] = padded

        def inv_zoom_out(prob, _t=top, _l=left, _sh=small_h, _sw=small_w):
            cropped = prob[_t:_t + _sh, _l:_l + _sw]
            return F.interpolate(
                cropped[None, None].float(), size=(img_h, img_w),
                mode='bilinear', align_corners=False,
            )[0, 0]

        return imgs_aug, inv_zoom_out


# ---------------------------------------------------------------------------
# Module B — Support permutation ensembling
# ---------------------------------------------------------------------------

def permute_supports(
    support_imgs: torch.Tensor,
    support_masks: torch.Tensor,
    perm: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder support frames along the shot dimension."""
    idx = torch.tensor(perm, dtype=torch.long)
    return support_imgs[:, idx], support_masks[:, idx]


def get_permutations(n_shots: int, n_perm: int, seed: int = 42) -> List[List[int]]:
    """
    Return up to n_perm distinct permutations of range(n_shots).

    The identity permutation is always first.
    When n_shots == 1, always returns [[0]] regardless of n_perm (no diversity
    possible; silently caps at 1 pass).
    """
    identity = list(range(n_shots))
    if n_shots == 1 or n_perm <= 1:
        return [identity]
    rng = np.random.default_rng(seed)
    perms: List[List[int]] = [identity]
    seen = {tuple(identity)}
    for _ in range(n_perm - 1):
        for _ in range(200):  # bounded attempts to find a new distinct permutation
            p = rng.permutation(n_shots).tolist()
            key = tuple(p)
            if key not in seen:
                perms.append(p)
                seen.add(key)
                break
    return perms


# ---------------------------------------------------------------------------
# Module C — Post-processing
# ---------------------------------------------------------------------------

def drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components with fewer than min_area pixels.
    No-op when min_area <= 0.
    """
    if min_area <= 0:
        return mask
    labeled, n_components = label(mask)
    out = np.zeros_like(mask)
    for comp in range(1, n_components + 1):
        if (labeled == comp).sum() >= min_area:
            out[labeled == comp] = 1
    return out.astype(mask.dtype)


def binarize(
    prob: torch.Tensor,
    base_threshold: float,
    mode: str,
    conf_floor: float = 0.3,
) -> torch.Tensor:
    """
    Binarize a probability map [H, W] (float in [0, 1]).

    mode='fixed': threshold at base_threshold.
    mode='otsu':  compute Otsu threshold on the probability values.
        Falls back to fixed when max(prob) < conf_floor to avoid
        over-segmenting empty-query episodes.

    Returns a bool tensor [H, W] on the same device as prob.
    """
    if mode == 'fixed' or prob.max().item() < conf_floor:
        return prob > base_threshold
    prob_np = prob.cpu().float().numpy()
    thresh = _otsu_threshold(prob_np)
    return torch.from_numpy(prob_np > thresh).to(prob.device)


def _otsu_threshold(arr: np.ndarray) -> float:
    """Otsu threshold without a scikit-image dependency."""
    hist, bin_edges = np.histogram(arr.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cum_hist = np.cumsum(hist)
    cum_val = np.cumsum(hist * bin_centers)
    w0 = cum_hist / total
    w1 = 1.0 - w0
    mu0 = np.where(cum_hist > 0, cum_val / np.maximum(cum_hist, 1e-10), 0.0)
    total_mean = (hist * bin_centers).sum() / total
    mu1 = np.where(
        (total - cum_hist) > 0,
        (total_mean * total - cum_val) / np.maximum(total - cum_hist, 1e-10),
        0.0,
    )
    with np.errstate(invalid='ignore'):
        sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b = np.nan_to_num(sigma_b)
    return float(bin_centers[int(np.argmax(sigma_b))])
