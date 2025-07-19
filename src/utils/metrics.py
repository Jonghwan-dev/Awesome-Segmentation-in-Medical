# utils/metrics.py
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import binary_erosion
from torch import Tensor


# --------------------------------------------------------------------------
# --- 핵심 분할 메트릭 (Core Segmentation Metrics) ---
# --------------------------------------------------------------------------

def dice_score(pred: Tensor, target: Tensor, eps: float = 1e-7) -> float:
    """배치에 대한 평균 Dice 계수를 계산합니다."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()

def iou_score(pred: Tensor, target: Tensor, eps: float = 1e-7) -> float:
    """배치에 대한 평균 IoU (Jaccard) 점수를 계산합니다."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def hd95_batch(preds: Tensor, targets: Tensor) -> float:
    """
    배치에 대한 평균 95th percentile Hausdorff Distance를 안정적으로 계산합니다.
    내부적으로 hd95 함수를 호출하고, 유효한 값들만으로 평균을 냅니다.
    'hd95_batch'라는 이름은 이 함수가 단일 이미지가 아닌 '배치' 전체의 평균을 계산한다는 의미입니다.
    """
    hd95_values = [
        hd95(p.squeeze().cpu().numpy() > 0.5, t.squeeze().cpu().numpy() > 0.5)
        for p, t in zip(preds, targets)
    ]
    
    # NaN 값을 명시적으로 제외하고 유효한 값만으로 리스트를 만듭니다.
    valid_hd95s = [v for v in hd95_values if not np.isnan(v)]
    
    # 만약 유효한 값이 하나도 없다면 페널티 값을 반환하고, 그렇지 않으면 평균을 반환합니다.
    if not valid_hd95s:
        return 999.0
    else:
        return np.mean(valid_hd95s)

# --------------------------------------------------------------------------
# --- 픽셀 단위 및 분류 스타일 메트릭 ---
# --------------------------------------------------------------------------

def pixel_accuracy(pred: Tensor, target: Tensor) -> float:
    """전체 픽셀에 대한 정확도를 계산합니다."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()

# ... (sensitivity, specificity, precision 등 다른 메트릭 함수들은 그대로 유지) ...

# --------------------------------------------------------------------------
# --- 보조 함수 (Helper Functions) ---
# --------------------------------------------------------------------------

def get_boundary(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """이진 침식(erosion)을 사용하여 마스크의 경계선을 찾습니다."""
    eroded = binary_erosion(mask, footprint=np.ones((3, 3)), connectivity=connectivity)
    boundary = np.logical_xor(mask, eroded)
    return boundary

def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    단일 마스크 쌍에 대한 95th percentile Hausdorff Distance를 안정적으로 계산합니다.
    어떤 오류가 발생하더라도 NaN 대신 페널티 값을 반환하여 랭킹 시스템의 오작동을 방지합니다.
    """
    try:
        pred_bool = pred.astype(bool)
        gt_bool = gt.astype(bool)
        
        # 경우 1: 두 마스크가 모두 비어있으면 거리는 0 (완벽하게 일치).
        if not np.any(pred_bool) and not np.any(gt_bool):
            return 0.0
        
        # 경우 2: 한쪽 마스크만 비어있으면, 이는 최악의 예측이므로 큰 페널티 값을 반환.
        if not np.any(pred_bool) or not np.any(gt_bool):
            return 999.0

        # 경계선 위의 점들 좌표를 찾음
        pred_pts = np.argwhere(get_boundary(pred_bool))
        gt_pts = np.argwhere(get_boundary(gt_bool))

        # 만약 경계선이 없으면 (예: 1픽셀짜리 점), 마스크 전체 점을 사용
        if len(pred_pts) == 0: pred_pts = np.argwhere(pred_bool)
        if len(gt_pts) == 0: gt_pts = np.argwhere(gt_bool)
        
        # 그래도 점이 없으면 페널티 값 반환
        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return 999.0

        # Hausdorff Distance 계산
        dists_pred_to_gt = np.min(np.linalg.norm(pred_pts[:, None] - gt_pts[None, :], axis=2), axis=1)
        dists_gt_to_pred = np.min(np.linalg.norm(gt_pts[:, None] - pred_pts[None, :], axis=2), axis=1)
        
        return np.percentile(np.hstack([dists_pred_to_gt, dists_gt_to_pred]), 95)
    except Exception:
        # 계산 중 어떤 종류의 오류가 발생하더라도 페널티 값을 반환
        return 999.0
