# src/utils/metrics.py
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import binary_erosion
from torch import Tensor

# --- Core Segmentation Metrics (Returning sum and count for proper averaging) ---
# This part is already correct. No changes needed.
def dice_score(pred: Tensor, target: Tensor, eps: float = 1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return torch.sum(dice).item(), pred.size(0)

def iou_score(pred: Tensor, target: Tensor, eps: float = 1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return torch.sum(iou).item(), pred.size(0)

def pixel_accuracy(pred: Tensor, target: Tensor):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return correct.item(), total

def hd95_batch(preds: Tensor, targets: Tensor):
    hd95_values = [
        hd95(p.squeeze().cpu().numpy() > 0.5, t.squeeze().cpu().numpy() > 0.5)
        for p, t in zip(preds, targets)
    ]
    valid_hd95s = [v for v in hd95_values if not np.isnan(v)]
    if not valid_hd95s:
        return 0.0, 0
    else:
        return np.sum(valid_hd95s), len(valid_hd95s)


# --- Helper Functions ---
def get_boundary(mask: np.ndarray) -> np.ndarray:
    footprint = np.ones((3, 3)) 
    eroded = binary_erosion(mask, footprint=footprint)
    return np.logical_xor(mask, eroded)

def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bool, gt_bool = pred.astype(bool), gt.astype(bool)

    if not np.any(gt_bool) and not np.any(pred_bool): return 0.0
    if not np.any(gt_bool) and np.any(pred_bool): return np.nan
    if np.any(gt_bool) and not np.any(pred_bool):
        return np.sqrt(gt.shape[0]**2 + gt.shape[1]**2)
        
    try:
        pred_pts = np.argwhere(get_boundary(pred_bool))
        gt_pts = np.argwhere(get_boundary(gt_bool))

        if pred_pts.shape[0] == 0: pred_pts = np.argwhere(pred_bool)
        if gt_pts.shape[0] == 0: gt_pts = np.argwhere(gt_bool)
        
        # This is the crucial part that was causing the error.
        # We need to compute the distances manually, not use directed_hausdorff.
        
        # For each point in pred_pts, find the minimum distance to gt_pts
        dists_pred_to_gt = np.min(np.linalg.norm(pred_pts[:, None] - gt_pts[None, :], axis=2), axis=1)
        
        # For each point in gt_pts, find the minimum distance to pred_pts
        dists_gt_to_pred = np.min(np.linalg.norm(gt_pts[:, None] - pred_pts[None, :], axis=2), axis=1)

        # The 95th percentile is calculated from all these distances
        return np.percentile(np.concatenate([dists_pred_to_gt, dists_gt_to_pred]), 95)

    except Exception:
        # Fallback for any other unexpected errors
        return np.sqrt(gt.shape[0]**2 + gt.shape[1]**2)