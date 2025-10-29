# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing
    on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class MGIoU2DLoss(nn.Module):
    """MGIoU loss for either rotated rectangles (x,y,w,h,θ) or explicit 4-corner boxes."""
    def __init__(
        self,
        representation: str = "rect",        # "rect" or "corner"
        reduction: str = "mean",
        loss_weight: float = 1.0,
        fast_mode: bool = False,
    ):
        super().__init__()
        if representation not in {"rect", "corner"}:
            raise ValueError("representation must be 'rect' or 'corner'")
        self.representation = representation
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.fast_mode = fast_mode

        # only needed if converting from (x,y,w,h,θ)
        self.register_buffer(
            "_unit_square",
            torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float32),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        red = reduction_override or self.reduction

        # --- convert or validate inputs ---
        if self.representation == "rect":
            # (B,5) → corners
            if pred.shape != target.shape or pred.shape[-1] != 5:
                raise ValueError("Expected (B,5) boxes for 'rect' mode")
            B = pred.size(0)

            # detect degenerate GTs → fallback to L1 as before
            all_zero = (target.abs().sum(dim=1) == 0)
            losses = pred.new_zeros(B)
            if all_zero.any():
                l1 = F.l1_loss(pred[all_zero], target[all_zero], reduction="none")
                losses[all_zero] = l1.sum(dim=1)

            mask = ~all_zero
            if mask.any():
                # convert to corners
                c1 = self._rect_to_corners(pred[mask])
                c2 = self._rect_to_corners(target[mask])
                losses[mask] = self._mgiou_boxes(c1, c2)

        else:  # "corner" mode
            # expect (B,4,2)
            if pred.ndim != 3 or pred.shape[-2:] != (4, 2):
                raise ValueError("Expected (B,4,2) corners for 'corner' mode")
            if pred.shape != target.shape:
                raise ValueError("pred and target must match shape")
            B = pred.size(0)
            # compute MGIoU on all
            losses = self._mgiou_boxes(pred, target)

        # --- weighting & reduction (common) ---
        if weight is not None:
            weight = weight.view(-1) if weight.dim() > 1 else weight
            losses = losses * weight
            if avg_factor is None:
                avg_factor = weight.sum().clamp_min(1.0)
        avg_factor = float(avg_factor or B)
        loss = self._reduce(losses, red) / avg_factor
        return (loss * self.loss_weight).to(pred.dtype)

    def _mgiou_boxes(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        # c1, c2: (N,4,2)
        axes = torch.cat((self._rect_axes(c1), self._rect_axes(c2)), dim=1)  # [N,4,2]
        proj1 = c1.to(axes.dtype) @ axes.transpose(1, 2)  # [N,4,4]
        proj2 = c2.to(axes.dtype) @ axes.transpose(1, 2)

        mn1, mx1 = proj1.min(dim=1).values, proj1.max(dim=1).values
        mn2, mx2 = proj2.min(dim=1).values, proj2.max(dim=1).values

        if self.fast_mode:
            num = torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)
            den = torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2) + 0 # ESPilon to avoid zero division
            giou1d = num / den
        else:
            inter = (torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)).clamp(min=0.0)
            union = (mx1 - mn1) + (mx2 - mn2) - inter
            hull  = (torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2))
            giou1d = inter / union - (hull - union) / hull

        return ((1.0 - giou1d.mean(dim=-1)) * 0.5)

    def _rect_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        trans, wh, angle = boxes[:, :2], boxes[:, 2:4], boxes[:, 4]
        base = self._unit_square.to(boxes.dtype).unsqueeze(0) * (wh * 0.5).unsqueeze(1)  # [B,4,2]
        cos_a, sin_a = angle.cos(), angle.sin()
        rot = torch.stack(
            (torch.stack((cos_a, -sin_a), -1), torch.stack((sin_a, cos_a), -1)),
            dim=1,
        )  # [B,2,2]
        return torch.bmm(base, rot) + trans.unsqueeze(1)  # [B,4,2]

    @staticmethod
    def _rect_axes(corners: torch.Tensor) -> torch.Tensor:
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 3] - corners[:, 0]
        normals = torch.stack((-e1[..., 1:], e1[..., :1], -e2[..., 1:], e2[..., :1]), dim=1)
        return normals.view(-1, 2, 2)

    @staticmethod
    def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {reduction!r}")

class MGIoU2DPlus(nn.Module):
    """MGIoU for arbitrary convex quadrangles with an optional convexity loss."""

    def __init__(
        self,
        convex_weight: float = 0.0,
        fast_mode: bool = False,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'.")
        self.convex_weight = convex_weight
        self.fast_mode = fast_mode
        self.reduction = reduction

    # ------------------------------------------------------------------ #
    #                               API                                   #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        pred: torch.Tensor,                  # [B,N,2] where N≥3
        target: torch.Tensor,                # [B,N,2] where N≥3
        visible_mask: torch.Tensor | None = None,
        reduction_override: str | None = None,
    ) -> torch.Tensor:
        """
        Compute per‐sample MGIoU + convexity, then reduce.
        
        Important: pred and target must have the same N (number of corners) within a batch.
        If polygons have different corner counts, pad to max with repeated last corner before calling.
        
        Args:
            pred: Predicted polygons of shape [B, N, 2] where N≥3
            target: Target polygons of shape [B, N, 2] where N≥3
            visible_mask: Optional mask to filter samples
            reduction_override: Override reduction mode
            
        Returns:
            Loss value (scalar or per-sample depending on reduction)
        """
        if visible_mask is not None:
            mask = visible_mask.bool().squeeze()
            pred, target = pred[mask], target[mask]

        # per‐sample pure MGIoU
        losses = (1.0 - torch.vmap(self._mgiou_single)(pred, target)) * 0.5

        # add convex‐penalty if requested
        if self.convex_weight != 0.0:
            convex_loss = self._convexity_loss(pred)  # [B]
            losses = losses + self.convex_weight * convex_loss

        # reduce
        red = reduction_override or self.reduction
        return self._reduce(losses, red)

    @staticmethod
    def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    # ------------------------------------------------------------------ #
    #                       MGIoU internals                              #
    # ------------------------------------------------------------------ #
    def _mgiou_single(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        axes = torch.cat((self._candidate_axes(c1), self._candidate_axes(c2)))
        giou1d = []
        for axis in axes:
            min1, max1 = (c1 @ axis).min(), (c1 @ axis).max()
            min2, max2 = (c2 @ axis).min(), (c2 @ axis).max()
            if self.fast_mode:
                numerator = torch.minimum(max1, max2) - torch.maximum(min1, min2)
                denominator = torch.maximum(max1, max2) - torch.minimum(min1, min2)
                giou1d.append(numerator / denominator)
            else:
                # Get intersection, union, and convex hull, then compute MGIoU
                inter = (torch.minimum(max1, max2) - torch.maximum(min1, min2)).clamp(min=0.0)
                union = (max1 - min1) + (max2 - min2) - inter
                hull = (torch.maximum(max1, max2) - torch.minimum(min1, min2))
                giou1d.append(inter / union - (hull - union) / hull)
        return torch.mean(torch.stack(giou1d))

    @staticmethod
    def _candidate_axes(corners: torch.Tensor) -> torch.Tensor:
        center = corners.mean(dim=0, keepdim=True)
        angles = torch.atan2(corners[:, 1] - center[0, 1], corners[:, 0] - center[0, 0])
        corners = corners[angles.argsort()]  # clockwise
        edges = torch.vstack((corners[1:] - corners[:-1], corners[:1] - corners[-1:]))
        normals = torch.stack((edges[:, 1], -edges[:, 0]), dim=1)
        return normals

    # ------------------------------------------------------------------ #
    #                    Convexity consistency penalty                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convexity_loss(polygons: torch.Tensor) -> torch.Tensor:
        """Mean ε where ε>0 indicates non-convex vertices (0 = perfectly convex)."""
        B, N, _ = polygons.shape  # N=4 for quadrangles but works for any N≥3
        v_prev = polygons[:, torch.arange(N) - 1]          # i-1
        v_curr = polygons
        v_next = polygons[:, (torch.arange(N) + 1) % N]    # i+1

        edge1 = v_prev - v_curr
        edge2 = v_next - v_curr
        cross = edge1[..., 0] * edge2[..., 1] - edge1[..., 1] * edge2[..., 0]  # (B,N)

        sign_ref = torch.where(cross[:, 0:1].abs() <= 0, torch.ones_like(cross[:, 0:1]), cross[:, 0:1]).sign()
        penalty = torch.clamp(-sign_ref * cross, min=0.0)  # negative => non-convex
        return penalty.mean(dim=1)  # [B]

class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16, use_mgiou: bool = False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.mgiou_loss = MGIoU2DLoss(representation="corner", reduction="sum") if use_mgiou else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        if self.mgiou_loss:
            # Convert xyxy boxes to 4-corner format for MGIoU
            pred_corners = self._xyxy_to_corners(pred_bboxes[fg_mask])
            target_corners = self._xyxy_to_corners(target_bboxes[fg_mask])
            loss_iou = self.mgiou_loss(
                pred_corners,
                target_corners.to(pred_corners.dtype),
                weight=weight.to(pred_corners.dtype),
                avg_factor=target_scores_sum
            )
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _xyxy_to_corners(boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert xyxy bounding boxes to 4-corner representation.
        
        Args:
            boxes (torch.Tensor): Boxes in xyxy format, shape (N, 4).
        
        Returns:
            (torch.Tensor): Boxes as 4 corners, shape (N, 4, 2).
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        corners = torch.stack([
            torch.stack([x1, y1], dim=-1),  # top-left
            torch.stack([x2, y1], dim=-1),  # top-right
            torch.stack([x2, y2], dim=-1),  # bottom-right
            torch.stack([x1, y2], dim=-1),  # bottom-left
        ], dim=1)
        return corners

class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int, use_mgiou: bool = False):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_mgiou=False)  # Don't use parent's MGIoU for axis-aligned boxes
        self.mgiou_loss = MGIoU2DLoss(representation="rect", reduction="sum") if use_mgiou else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        if self.mgiou_loss:
            loss_iou = self.mgiou_loss(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask].to(pred_bboxes.dtype),
                weight=weight.to(pred_bboxes.dtype),
                avg_factor=target_scores_sum
            )
        else:
            iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10, use_mgiou: bool = False):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max, use_mgiou=use_mgiou).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, use_mgiou: bool = False, only_mgiou: bool = False):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, use_mgiou=use_mgiou)  # Pass use_mgiou to parent for bbox loss
        self.overlap = model.args.overlap_mask
        self.mgiou_loss = MGIoU2DPlus(reduction="sum", convex_weight=0.1) if use_mgiou else None
        self.use_mgiou = use_mgiou
        self.only_mgiou = only_mgiou and use_mgiou  # only_mgiou requires use_mgiou to be True
        
        # Enhanced loss weights for hybrid approach
        # Adjusted for proper scale: Chamfer ~0.1-0.3, MGIoU ~0.3-0.7 for untrained models
        self.mgiou_weight = 2.0      # MGIoU for IoU-based shape matching (was 0.4)
        self.chamfer_weight = 3.0    # Chamfer for point-to-point matching (was 0.5) - increased since it's main signal
        self.corner_penalty_weight = 0.1  # Soft penalty for corner count mismatch (kept same, currently unused)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        # Loss tensor layout changes based on use_mgiou and only_mgiou:
        # Normal (no MGIoU):     [box, seg, cls, dfl] (4 elements)
        # With MGIoU:            [box, seg, cls, dfl, mgiou, chamfer, corner_penalty] (7 elements)
        # only_mgiou=True:       [box, cls, dfl, mgiou, chamfer, corner_penalty] (6 elements, seg_loss omitted)
        if self.only_mgiou:
            loss = torch.zeros(6, device=self.device)  # box, cls, dfl, mgiou, chamfer, corner_penalty
            cls_idx, dfl_idx = 1, 2
            mgiou_idx, chamfer_idx, corner_idx = 3, 4, 5
        elif self.use_mgiou:
            loss = torch.zeros(7, device=self.device)  # box, seg, cls, dfl, mgiou, chamfer, corner_penalty
            cls_idx, dfl_idx = 2, 3
            mgiou_idx, chamfer_idx, corner_idx = 4, 5, 6
        else:
            loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
            cls_idx, dfl_idx = 2, 3
        
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[cls_idx] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[dfl_idx] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            if self.only_mgiou:
                # Only compute mgiou_loss, skip seg_loss
                # Returns: (seg_loss, mgiou_loss, chamfer_loss, corner_penalty)
                _, loss[mgiou_idx], loss[chamfer_idx], loss[corner_idx] = self.calculate_segmentation_loss(
                    fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
                )
            elif self.use_mgiou:
                # Compute both seg_loss and mgiou_loss components
                # Returns: (seg_loss, mgiou_loss, chamfer_loss, corner_penalty)
                loss[1], loss[mgiou_idx], loss[chamfer_idx], loss[corner_idx] = self.calculate_segmentation_loss(
                    fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
                )
            else:
                # Only compute seg_loss
                loss[1] = self.calculate_segmentation_loss(
                    fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
                )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            if not self.only_mgiou:
                loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        if not self.only_mgiou:
            loss[1] *= self.hyp.box  # seg gain
        loss[cls_idx] *= self.hyp.cls  # cls gain
        loss[dfl_idx] *= self.hyp.dfl  # dfl gain
        if self.use_mgiou:
            # Apply weights to separate MGIoU components
            loss[mgiou_idx] *= self.hyp.box * self.mgiou_weight  # mgiou component
            loss[chamfer_idx] *= self.hyp.box * self.chamfer_weight  # chamfer component
            loss[corner_idx] *= self.hyp.box * self.corner_penalty_weight  # corner penalty component

        return loss * batch_size, loss.detach()  # loss(box, seg/cls, cls/dfl, dfl/mgiou, [mgiou])

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    @staticmethod
    def interpolate_polygon_padding(corners: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Pad polygon to target size by interpolating between existing corners.
        This creates smooth transitions instead of degenerate repeated corners.
        
        Args:
            corners (torch.Tensor): Input corners of shape (N, 2) where N >= 3.
            target_size (int): Target number of corners (must be >= N).
            
        Returns:
            (torch.Tensor): Padded corners of shape (target_size, 2).
        """
        n = corners.shape[0]
        if n >= target_size:
            return corners
        
        # Calculate how many points to insert
        num_to_insert = target_size - n
        
        # Calculate edge lengths to distribute insertions proportionally
        edges = torch.roll(corners, -1, dims=0) - corners
        edge_lengths = torch.norm(edges, dim=1)
        
        # Distribute insertions proportionally to edge lengths
        # Longer edges get more inserted points
        insertions_per_edge = (edge_lengths / edge_lengths.sum() * num_to_insert).round().int()
        
        # Adjust to ensure exact count
        diff = num_to_insert - insertions_per_edge.sum().item()
        if diff > 0:
            # Add remaining to longest edges
            longest_edges = torch.argsort(edge_lengths, descending=True)[:diff]
            insertions_per_edge[longest_edges] += 1
        elif diff < 0:
            # Remove from edges that have insertions
            for _ in range(-diff):
                idx = (insertions_per_edge > 0).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    insertions_per_edge[idx[0]] -= 1
        
        # Build new corner list with interpolated points
        new_corners = []
        for i in range(n):
            new_corners.append(corners[i])
            num_insert = insertions_per_edge[i].item()
            if num_insert > 0:
                next_corner = corners[(i + 1) % n]
                for j in range(1, num_insert + 1):
                    alpha = j / (num_insert + 1)
                    interpolated = corners[i] * (1 - alpha) + next_corner * alpha
                    new_corners.append(interpolated)
        
        return torch.stack(new_corners)

    @staticmethod
    def chamfer_distance(pred_corners: torch.Tensor, gt_corners: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional Chamfer distance between two point sets.
        More stable than MGIoU for gradient flow.
        
        Args:
            pred_corners (torch.Tensor): Predicted corners of shape (N1, 2).
            gt_corners (torch.Tensor): Ground truth corners of shape (N2, 2).
            
        Returns:
            (torch.Tensor): Chamfer distance (scalar) in range [0, ~1.414] for normalized coords.
        """
        # pred -> gt: for each pred point, find nearest gt point
        pred_exp = pred_corners.unsqueeze(1)  # (N1, 1, 2)
        gt_exp = gt_corners.unsqueeze(0)      # (1, N2, 2)
        dist_matrix = torch.sum((pred_exp - gt_exp) ** 2, dim=2)  # (N1, N2) - squared distances
        
        # Take sqrt for actual Euclidean distance (not squared)
        # This gives more interpretable loss values in [0, sqrt(2)] range
        pred_to_gt = dist_matrix.min(dim=1)[0].sqrt().mean()  # (N1,) -> scalar
        gt_to_pred = dist_matrix.min(dim=0)[0].sqrt().mean()  # (N2,) -> scalar
        
        return (pred_to_gt + gt_to_pred) / 2

    @staticmethod
    def smooth_l1_corner_penalty(pred_count: int, gt_count: int, tolerance: int = 2, beta: float = 2.0) -> float:
        """
        Soft penalty for corner count mismatch with tolerance.
        No penalty if difference is within tolerance, smooth penalty beyond.
        
        Args:
            pred_count (int): Number of predicted corners.
            gt_count (int): Number of ground truth corners.
            tolerance (int): Acceptable corner count difference (no penalty within this range).
            beta (float): Smooth L1 beta parameter (controls smoothness).
            
        Returns:
            (float): Corner count penalty (scalar value, will be converted to tensor by caller).
        """
        diff = abs(pred_count - gt_count)
        if diff <= tolerance:
            return 0.0
        
        # Apply smooth L1 only to the excess beyond tolerance
        excess = diff - tolerance
        if excess < beta:
            penalty = 0.5 * (excess ** 2) / beta
        else:
            penalty = excess - 0.5 * beta
        
        return float(penalty)

    @staticmethod
    def mask_to_polygon_corners(mask: torch.Tensor, epsilon_factor: float = 0.02, threshold: float = 0.4) -> torch.Tensor | None:
        """
        Convert a binary mask to polygon corners using OpenCV contour approximation.
        
        Enhanced version with:
        - Lower threshold (0.4) for softer masks
        - Coordinate normalization to [0, 1]
        - Improved stability
        
        Note: MGIoU2DPlus supports any N≥3 corners, so we use a simple one-shot approximation
        instead of binary search. This is faster and the exact corner count doesn't matter.

        Args:
            mask (torch.Tensor): Binary mask of shape (H, W).
            epsilon_factor (float): Approximation accuracy factor (0.01-0.05 typical). 
                                   Lower = more corners, Higher = fewer corners.
            threshold (float): Threshold for binarization (0.4 default, softer than 0.5).

        Returns:
            (torch.Tensor | None): Normalized polygon corners of shape (N, 2) where N≥3 
                                   with coordinates in [0, 1], or None if extraction fails.
        """
        try:
            # Convert to numpy with lower threshold for softer masks
            mask_np = (mask.detach().cpu().numpy() > threshold).astype('uint8')
            h, w = mask_np.shape
            
            # Find contours
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            # Get largest contour by area
            contour = max(contours, key=cv2.contourArea)
            
            # Minimum 3 corners required for MGIoU2DPlus
            if len(contour) < 3:
                return None
            
            # Simple one-shot polygon approximation
            arc_len = cv2.arcLength(contour, True)
            epsilon = epsilon_factor * arc_len
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If too few corners, try with convex hull
            if len(approx) < 3:
                approx = cv2.convexHull(contour)
                if len(approx) < 3:
                    return None
            
            # Convert back to torch tensor (shape: [N, 2] where N≥3)
            corners = torch.from_numpy(approx.reshape(-1, 2)).float().to(mask.device)

            # Remove duplicate points while preserving original occurrence order
            if corners.shape[0] > 1:
                try:
                    uniq_idx = np.unique(corners.cpu().numpy(), axis=0, return_index=True)[1]
                    uniq_idx.sort()
                    corners = corners[uniq_idx]
                except Exception:
                    # Fallback: keep corners as-is if unique fails for any reason
                    pass

            # Need at least 3 unique corners
            if corners.shape[0] < 3:
                return None

            # Clamp to mask bounds and round to integer pixel coords for stability
            corners[:, 0] = corners[:, 0].clamp(0, w - 1)
            corners[:, 1] = corners[:, 1].clamp(0, h - 1)
            corners = corners.round()

            # Deterministic ordering: sort points by angle around centroid
            centroid = corners.mean(dim=0)
            angles = torch.atan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
            order = torch.argsort(angles)
            corners = corners[order]

            # Rotate so that the corner with smallest (y + x) (top-left-ish) is first
            start_idx = torch.argmin(corners[:, 0] + corners[:, 1])
            corners = torch.roll(corners, -int(start_idx), dims=0)

            # Normalize coordinates to [0, 1] for scale-invariance
            corners[:, 0] = corners[:, 0] / (w - 1)
            corners[:, 1] = corners[:, 1] / (h - 1)

            return corners
            
        except (RuntimeError, ValueError, cv2.error):
            return None

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor | tuple): If use_mgiou is False, returns the seg_loss only.
                If use_mgiou is True, returns a tuple of (seg_loss, mgiou_loss, chamfer_loss, corner_penalty).

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        
        # Initialize components as tensors on the correct device
        device = proto.device
        mgiou_component = torch.tensor(0.0, device=device)
        chamfer_component = torch.tensor(0.0, device=device)
        corner_penalty_component = torch.tensor(0.0, device=device)

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                # Standard BCE mask loss (skip if only_mgiou is True)
                if not self.only_mgiou:
                    loss += self.single_mask_loss(
                        gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                    )

                # MGIoU polygon loss (if enabled) - returns separate components
                if self.use_mgiou and self.mgiou_loss is not None:
                    mgiou, chamfer, corner_penalty = self.compute_mgiou_mask_loss(
                        gt_mask, pred_masks_i[fg_mask_i], proto_i, fg_mask_i.sum()
                    )
                    mgiou_component += mgiou
                    chamfer_component += chamfer
                    corner_penalty_component += corner_penalty

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        total_loss = loss / fg_mask.sum()
        
        # Return separate losses if MGIoU is enabled
        if self.use_mgiou:
            # Normalize by number of foreground masks, ensure proper device handling
            fg_sum = fg_mask.sum()
            mgiou_normalized = mgiou_component / fg_sum if mgiou_component.item() > 0 else torch.tensor(0.0, device=total_loss.device)
            chamfer_normalized = chamfer_component / fg_sum if chamfer_component.item() > 0 else torch.tensor(0.0, device=total_loss.device)
            corner_normalized = corner_penalty_component / fg_sum if corner_penalty_component.item() > 0 else torch.tensor(0.0, device=total_loss.device)
            return total_loss, mgiou_normalized, chamfer_normalized, corner_normalized
        
        return total_loss

    def compute_mgiou_mask_loss(
        self,
        gt_masks: torch.Tensor,
        pred_coeffs: torch.Tensor,
        proto: torch.Tensor,
        num_instances: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute enhanced hybrid MGIoU loss for mask polygons.
        
        Uses combination of:
        1. MGIoU for IoU-based shape matching
        2. Chamfer distance for point-to-point matching  
        3. Soft corner count penalty
        
        This approach maintains flexible corner counts while providing stable gradients.

        Args:
            gt_masks (torch.Tensor): Ground truth masks of shape (N, H, W).
            pred_coeffs (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            num_instances (int): Number of instances for normalization.

        Returns:
            (tuple): (mgiou_loss, chamfer_loss, corner_penalty) - separate loss components for logging.
        """
        zero_tensor = torch.tensor(0.0, device=gt_masks.device)
        
        if num_instances == 0:
            return zero_tensor, zero_tensor, zero_tensor

        # Generate predicted masks
        pred_masks = torch.einsum("in,nhw->ihw", pred_coeffs, proto)  # (N, H, W)
        pred_masks = pred_masks.sigmoid()  # Apply sigmoid to get probabilities

        # Convert masks to normalized polygon corners
        pred_polygons = []
        gt_polygons = []
        pred_counts = []
        gt_counts = []
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            # Convert masks to normalized polygons (coordinates in [0, 1])
            pred_poly = self.mask_to_polygon_corners(pred_mask)
            gt_poly = self.mask_to_polygon_corners(gt_mask)
            
            if pred_poly is not None and gt_poly is not None:
                pred_polygons.append(pred_poly)
                gt_polygons.append(gt_poly)
                pred_counts.append(pred_poly.shape[0])
                gt_counts.append(gt_poly.shape[0])

        # If no valid polygons, return zero losses
        if len(pred_polygons) == 0:
            return zero_tensor, zero_tensor, zero_tensor

        # Compute three loss components
        # Initialize as tensors on the correct device
        total_mgiou_loss = torch.tensor(0.0, device=gt_masks.device)
        total_chamfer_loss = torch.tensor(0.0, device=gt_masks.device)
        total_corner_penalty = torch.tensor(0.0, device=gt_masks.device)
        
        for pred_poly, gt_poly, pred_n, gt_n in zip(pred_polygons, gt_polygons, pred_counts, gt_counts):
            # 1. Chamfer distance (most stable, direct point matching)
            chamfer = self.chamfer_distance(pred_poly, gt_poly)
            total_chamfer_loss += chamfer
            
            # 2. MGIoU loss (IoU-based shape matching)
            # Pad to same size using interpolation (not repetition!)
            max_corners = max(pred_n, gt_n)
            pred_padded = self.interpolate_polygon_padding(pred_poly, max_corners)
            gt_padded = self.interpolate_polygon_padding(gt_poly, max_corners)
            
            # Stack for batch processing
            pred_batch = pred_padded.unsqueeze(0)  # (1, max_corners, 2)
            gt_batch = gt_padded.unsqueeze(0)      # (1, max_corners, 2)
            
            # Compute MGIoU (returns loss, already 1 - IoU)
            mgiou = self.mgiou_loss(pred_batch, gt_batch)
            total_mgiou_loss += mgiou
            
            # 3. Soft corner count penalty (tolerance of ±2 corners)
            corner_penalty = self.smooth_l1_corner_penalty(pred_n, gt_n, tolerance=2, beta=2.0)
            # Convert float to tensor on correct device
            total_corner_penalty += corner_penalty

        # Average over all instances
        num_valid = len(pred_polygons)
        avg_mgiou = total_mgiou_loss / num_valid
        avg_chamfer = total_chamfer_loss / num_valid
        avg_corner_penalty = total_corner_penalty / num_valid
        
        # Return separate components (weights applied in __call__)
        return avg_mgiou, avg_chamfer, avg_corner_penalty


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, use_mgiou: bool = False):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, use_mgiou=use_mgiou)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, use_mgiou: bool = False):  # model must be de-paralleled
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max, use_mgiou=use_mgiou).to(self.device)
        self.use_mgiou = use_mgiou

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, use_mgiou: bool = False):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10, use_mgiou=use_mgiou)
        self.one2one = v8DetectionLoss(model, tal_topk=1, use_mgiou=use_mgiou)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, use_mgiou: bool = False):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, use_mgiou=use_mgiou)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, use_mgiou: bool = False):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model, use_mgiou=use_mgiou)
        self.vp_criterion = v8SegmentationLoss(model, use_mgiou=use_mgiou)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
