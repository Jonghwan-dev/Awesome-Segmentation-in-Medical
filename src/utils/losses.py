# utils/losses.py (수정된 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import numpy as np
from scipy.ndimage import distance_transform_edt as edt

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE
        return F_loss.mean()

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets):
        # loss를 device로 이동
        self.loss.pos_weight = self.loss.pos_weight.to(inputs.device)
        return self.loss(inputs, targets)

class EdgeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sobel = kornia.filters.Sobel()

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs)
        pred_edge = self.sobel(inputs_sigmoid)
        target_edge = self.sobel(targets)
        return F.l1_loss(pred_edge, target_edge)

def compute_sdf(mask):

    mask = mask.cpu().numpy()
    sdf = np.zeros_like(mask, dtype=np.float32)
    for b in range(mask.shape[0]):
        posmask = mask[b,0].astype(bool)
        if posmask.any(): # 마스크가 비어있지 않을 때만 계산
            negmask = ~posmask
            sdf[b,0] = edt(negmask) - edt(posmask)
    return torch.from_numpy(sdf)

class BoundaryLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, gt):

        pred_sigmoid = torch.sigmoid(pred_logits) 
        
        sdf_gt = compute_sdf(gt).to(gt.device)
        loss = torch.mean((pred_sigmoid - gt) * sdf_gt)
        return loss
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=0.5, bce_weight=0.5, pos_weight=1.0):
        super().__init__()
        self.weight = weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, inputs, targets, smooth=1e-7):
        
        self.bce.pos_weight = self.bce.pos_weight.to(inputs.device)

        # BCE loss
        bce_loss = self.bce(inputs, targets)

        # Dice loss
        inputs_sig = torch.sigmoid(inputs)
        # Flatten a N-D Tensor to 1-D Tensor
        intersection = (inputs_sig.flatten() * targets.flatten()).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sig.flatten().sum() + targets.flatten().sum() + smooth)
        
        return self.bce_weight * bce_loss + self.weight * dice_loss
    
class ComboLossHD(nn.Module):

    def __init__(self, alpha=0.8, gamma=2, edge_weight=1.0, boundary_weight=1.0, ce_weight=1.0, pos_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.edge = EdgeLoss()
        self.boundary = BoundaryLoss()
        self.ce = WeightedCrossEntropyLoss(pos_weight=pos_weight)
        self.edge_weight = edge_weight
        self.boundary_weight = boundary_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        edge_loss = self.edge(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        ce_loss = self.ce(inputs, targets)

        total_loss = (
            focal_loss
            + self.edge_weight * edge_loss
            + self.boundary_weight * boundary_loss
            + self.ce_weight * ce_loss
        )
        return total_loss