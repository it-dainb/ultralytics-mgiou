# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

_EPS = 1e-9

# Global flag for NaN debugging - can be enabled via environment variable or code
# Set ULTRALYTICS_DEBUG_NAN=1 to enable runtime NaN checks
import os
_DEBUG_NAN = os.environ.get("ULTRALYTICS_DEBUG_NAN", "0") == "1"

def _check_nan_tensor(tensor: torch.Tensor, name: str, location: str, extra_info: dict = None) -> None:
    """
    Debug utility to check for NaN/Inf values in tensors.
    Only runs when _DEBUG_NAN is True.
    
    Args:
        tensor: Tensor to check
        name: Name/description of the tensor
        location: Location in code where check is performed
        extra_info: Optional dict with additional debug information
    
    Raises:
        RuntimeError: If NaN or Inf detected and debug mode is enabled
    """
    if not _DEBUG_NAN:
        return
    
    if torch.isnan(tensor).any():
        nan_mask = torch.isnan(tensor)
        error_msg = (
            f"NaN detected in {name} at {location}\n"
            f"Shape: {tensor.shape}\n"
            f"NaN count: {nan_mask.sum().item()}"
        )
        
        # Add statistics for non-NaN values
        if not nan_mask.all():
            valid_vals = tensor[~nan_mask]
            error_msg += (
                f"\n\nValid value statistics:\n"
                f"  Min: {valid_vals.min().item():.6e}\n"
                f"  Max: {valid_vals.max().item():.6e}\n"
                f"  Mean: {valid_vals.mean().item():.6e}"
            )
        
        # Add extra debug info if provided
        if extra_info:
            error_msg += "\n\nAdditional debug info:"
            for key, val in extra_info.items():
                if torch.is_tensor(val):
                    error_msg += f"\n  {key}: shape={val.shape}, has_nan={torch.isnan(val).any().item()}"
                    if not torch.isnan(val).all():
                        valid = val[~torch.isnan(val)]
                        error_msg += f", min={valid.min().item():.6e}, max={valid.max().item():.6e}"
                else:
                    error_msg += f"\n  {key}: {val}"
        
        raise RuntimeError(error_msg)
    
    if torch.isinf(tensor).any():
        raise RuntimeError(
            f"Inf detected in {name} at {location}\n"
            f"Shape: {tensor.shape}\n"
            f"Inf count: {torch.isinf(tensor).sum().item()}"
        )

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

class MGIoURect(nn.Module):
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

        loss = self._reduce(losses, red)

        if avg_factor is not None:
            loss = loss / avg_factor
        
        return (loss * self.loss_weight).to(pred.dtype)

    def _mgiou_boxes(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        # c1, c2: (N,4,2)
        axes = torch.cat((self._rect_axes(c1), self._rect_axes(c2)), dim=1)  # [N,4,2]
        proj1 = c1.to(axes.dtype) @ axes.transpose(1, 2)  # [N,4,4]
        proj2 = c2.to(axes.dtype) @ axes.transpose(1, 2)

        mn1, mx1 = proj1.min(dim=1).values, proj1.max(dim=1).values
        mn2, mx2 = proj2.min(dim=1).values, proj2.max(dim=1).values

        inter = (torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)).clamp(min=0.0)
        hull  = (torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2))
        
        if self.fast_mode:
            giou1d = inter / (hull + _EPS)
        else:
            union = (mx1 - mn1) + (mx2 - mn2) - inter
            giou1d = inter / (union + _EPS) - (hull - union) / (hull + _EPS)

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

class MGIoUPoly(nn.Module):
    """
    MGIoU loss for general polygons with automatic degenerate edge detection.
    
    Supports:
    - Mixed vertex counts in batches (e.g., triangles, quads, hexagons together)
    - Automatic padding artifact detection and removal
    - Weighted loss computation
    - Flexible reduction strategies
    
    NaN Prevention Mechanisms:
    - Degenerate targets (all zeros) automatically fall back to L1 loss
    - Degenerate edges (repeated vertices from padding) are detected and masked out
    - Division operations use epsilon (_EPS = 1e-9) to prevent division by zero
    - Masked mean computation only averages over valid axes (non-degenerate edges)
    - Edge length detection filters out padding artifacts (edges < eps)
    - All clamp operations use safe minimum values
    
    Debugging:
    - Set environment variable ULTRALYTICS_DEBUG_NAN=1 to enable runtime NaN/Inf checks
    - Debug mode adds validation at critical points: inputs, GIoU computation, weighting, output
    - Raises RuntimeError with detailed diagnostics if NaN/Inf detected
    - Keep debug mode disabled in production for optimal performance
    """
    def __init__(self, fast_mode=False, reduction="mean", loss_weight=1.0, eps=1e-6):
        """
        Initialize MGIoUPoly with optimization options.
        
        Args:
            fast_mode: Use simplified GIoU computation (default: False)
            reduction: Loss reduction method: 'none', 'mean', or 'sum' (default: 'mean')
            loss_weight: Global weight for the loss (default: 1.0)
            eps: Threshold for detecting degenerate edges (default: 1e-6)
                 Edges with length < eps are considered padding artifacts
        """
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        
        self.fast_mode = fast_mode
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: float | None = None,
    ) -> torch.Tensor:
        """
        Compute MGIoU loss for polygon predictions.
        
        Args:
            pred: Predicted polygons [B, N, 2] where N is max vertices (padded if needed)
            target: Target polygons [B, M, 2] where M is max vertices (padded if needed)
            weight: Per-sample weights [B] or [B, 1] (optional)
            reduction_override: Override default reduction method (optional)
            avg_factor: Normalization factor for loss (optional, computed from weights if not provided)
            
        Returns:
            Computed loss value (scalar if reduction != 'none', else [B])
        
        Note:
            Set environment variable ULTRALYTICS_DEBUG_NAN=1 to enable runtime NaN checks.
        """
        # Debug checks for inputs (only runs if _DEBUG_NAN is True)
        _check_nan_tensor(pred, "pred", "MGIoUPoly.forward input")
        _check_nan_tensor(target, "target", "MGIoUPoly.forward input")
        if weight is not None:
            _check_nan_tensor(weight, "weight", "MGIoUPoly.forward input")
        
        B, N, _ = pred.shape
        M = target.shape[1]
        
        red = reduction_override or self.reduction
        
        # Detect completely degenerate targets (like MGIoURect strategy)
        # This handles cases where target is all zeros or invalid
        all_zero = (target.abs().sum(dim=(1, 2)) == 0)  # [B]
        losses = pred.new_zeros(B)
        
        # Fallback to L1 loss for degenerate targets
        if all_zero.any():
            # Flatten polygons for L1 loss
            pred_flat = pred[all_zero].view(all_zero.sum(), -1)
            target_flat = target[all_zero].view(all_zero.sum(), -1)
            l1 = F.l1_loss(pred_flat, target_flat, reduction="none")
            losses[all_zero] = l1.sum(dim=1).to(losses.dtype)
        
        # Compute MGIoU for valid targets
        valid_mask = ~all_zero
        if valid_mask.any():
            # Extract valid samples
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
            
            # Sort vertices by angle to ensure consistent ordering
            pred_sorted = self._sort_vertices(pred_valid)
            target_sorted = self._sort_vertices(target_valid)
            
            # Debug check for sorted vertices
            if _DEBUG_NAN:
                _check_nan_tensor(pred_sorted, "pred_sorted", "MGIoUPoly.forward after sorting")
                _check_nan_tensor(target_sorted, "target_sorted", "MGIoUPoly.forward after sorting")

            # Get axes and validity masks (detects degenerate edges from padding)
            axes1, mask1 = self._axes_with_mask(pred_sorted)
            axes2, mask2 = self._axes_with_mask(target_sorted)
            
            # Debug check for axes
            if _DEBUG_NAN:
                _check_nan_tensor(axes1, "axes1", "MGIoUPoly.forward axes computation")
                _check_nan_tensor(axes2, "axes2", "MGIoUPoly.forward axes computation")
            
            axes = torch.cat((axes1, axes2), dim=1)  # [B_valid, N+M, 2]
            mask = torch.cat((mask1, mask2), dim=1)  # [B_valid, N+M] - True = valid, False = degenerate

            # Project vertices onto all axes (vectorized)
            proj1 = torch.bmm(pred_sorted.to(axes.dtype), axes.transpose(1, 2))
            proj2 = torch.bmm(target_sorted.to(axes.dtype), axes.transpose(1, 2))
            
            # Safety: Replace NaN/Inf in projections with safe values
            # This can happen with extreme polygon coordinates
            proj1 = torch.where(torch.isnan(proj1) | torch.isinf(proj1), torch.zeros_like(proj1), proj1)
            proj2 = torch.where(torch.isnan(proj2) | torch.isinf(proj2), torch.zeros_like(proj2), proj2)
            
            min1, _ = proj1.min(dim=1)
            max1, _ = proj1.max(dim=1)
            min2, _ = proj2.min(dim=1)
            max2, _ = proj2.max(dim=1)
            
            # Debug check for projections
            if _DEBUG_NAN:
                _check_nan_tensor(proj1, "proj1", "MGIoUPoly.forward projection")
                _check_nan_tensor(proj2, "proj2", "MGIoUPoly.forward projection")
                _check_nan_tensor(min1, "min1", "MGIoUPoly.forward projection")
                _check_nan_tensor(max1, "max1", "MGIoUPoly.forward projection")
                _check_nan_tensor(min2, "min2", "MGIoUPoly.forward projection")
                _check_nan_tensor(max2, "max2", "MGIoUPoly.forward projection")

            # Compute GIoU on each axis
            # Note: _EPS prevents division by zero in edge cases
            inter = (torch.minimum(max1, max2) - torch.maximum(min1, min2)).clamp(min=0.0)
            hull  = torch.maximum(max1, max2) - torch.minimum(min1, min2)
            
            # Safety: Replace NaN values with safe defaults before clamping
            # NaN can occur from numerical issues in projections (e.g., Inf - Inf)
            inter = torch.where(torch.isnan(inter), torch.zeros_like(inter), inter)
            hull = torch.where(torch.isnan(hull), torch.full_like(hull, _EPS), hull)
            
            # Safety: Clamp hull to prevent division by very small values
            # Hull near zero indicates degenerate/collapsed polygons on this axis
            hull_safe = hull.clamp(min=_EPS)

            if self.fast_mode:
                # Simplified GIoU: intersection / hull
                giou1d = inter / hull_safe
            else:
                # Match reference formula exactly: inter/union - (hull-union)/hull
                union = (max1 - min1) + (max2 - min2) - inter
                
                # Safety: Clamp union to prevent negative or very small values
                union_safe = union.clamp(min=_EPS)
                
                # Compute GIoU components safely
                iou_term = inter / union_safe
                penalty_term = (hull_safe - union_safe) / hull_safe
                giou1d = iou_term - penalty_term
                
                # Additional safety: Clamp GIoU to valid range [-1, 1]
                # GIoU should theoretically be in [-1, 1], but numerical issues can push it outside
                giou1d = giou1d.clamp(min=-1.0, max=1.0)
            
            # Debug check giou1d right after computation (before masking)
            if _DEBUG_NAN:
                extra_pre = {
                    "inter": inter,
                    "hull": hull,
                    "hull_safe": hull_safe,
                    "min1": min1,
                    "max1": max1,
                    "min2": min2,
                    "max2": max2,
                }
                if not self.fast_mode:
                    extra_pre.update({
                        "union": union,
                        "union_safe": union_safe,
                        "iou_term": iou_term,
                        "penalty_term": penalty_term,
                    })
                _check_nan_tensor(giou1d, "giou1d", "MGIoUPoly.forward immediately after GIoU computation", extra_pre)

            # *** KEY: Masked mean - only average over valid (non-degenerate) axes ***
            # This allows correct batching of polygons with different vertex counts!
            
            # Safety: Replace any NaN in giou1d with 0.0 before masking
            # NaN can occur from numerical instability in edge cases, but should be masked out anyway
            giou1d = torch.where(torch.isnan(giou1d), torch.zeros_like(giou1d), giou1d)
            
            giou1d_masked = giou1d * mask.to(giou1d.dtype)  # zero out invalid axes
            num_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)  # avoid div by zero
            giou_val = giou1d_masked.sum(dim=1) / num_valid.squeeze()
            
            # Debug check for GIoU computation with detailed intermediate values
            if _DEBUG_NAN:
                extra_info = {
                    "inter": inter,
                    "hull": hull,
                    "hull_safe": hull_safe,
                    "giou1d": giou1d,
                    "giou1d_masked": giou1d_masked,
                    "num_valid": num_valid,
                    "mask": mask,
                }
                if not self.fast_mode:
                    extra_info.update({
                        "union": union,
                        "union_safe": union_safe,
                        "iou_term": iou_term,
                        "penalty_term": penalty_term,
                    })
                _check_nan_tensor(giou_val, "giou_val", "MGIoUPoly.forward after GIoU computation", extra_info)
            
            # Convert to loss: (1 - GIoU) / 2, range [0, 1]
            losses[valid_mask] = ((1.0 - giou_val) * 0.5).to(losses.dtype)
        
        # Debug check before weighting
        _check_nan_tensor(losses, "losses", "MGIoUPoly.forward before weighting")
        
        # --- weighting & reduction (matching MGIoURect behavior) ---
        if weight is not None:
            weight = weight.view(-1) if weight.dim() > 1 else weight
            losses = losses * weight
            
            if avg_factor is None:
                avg_factor = weight.sum().clamp_min(1.0)
        
        loss = self._reduce(losses, red)
        
        if avg_factor is not None:
            loss = loss / avg_factor
        
        # Debug check for final loss
        final_loss = (loss * self.loss_weight).to(pred.dtype)
        _check_nan_tensor(final_loss, "final_loss", "MGIoUPoly.forward output")
        
        return final_loss

    @staticmethod
    def _sort_vertices(poly):
        """Sort vertices by angle from centroid (like MGIoU2DPlus._candidate_axes)."""
        B, N, _ = poly.shape
        center = poly.mean(dim=1, keepdim=True)  # [B, 1, 2]
        angles = torch.atan2(poly[..., 1] - center[..., 1], poly[..., 0] - center[..., 0])  # [B, N]
        indices = angles.argsort(dim=1)  # [B, N]
        return torch.gather(poly, 1, indices.unsqueeze(-1).expand(-1, -1, 2))

    def _axes_with_mask(self, poly):
        """
        Compute edge normals and validity mask.
        
        Returns:
            normals: [B, N, 2] - edge normal vectors (normalized)
            mask: [B, N] - True if edge is valid (length > eps), False if degenerate
        
        This automatically detects padding artifacts (repeated vertices) and excludes
        their axes from the mean calculation, enabling correct mixed-vertex batching.
        """
        edges = poly.roll(-1, dims=1) - poly  # [B, N, 2]
        edge_lengths = torch.norm(edges, dim=-1, keepdim=True)  # [B, N, 1]
        
        # Mark edges as valid if length > eps
        # Degenerate edges (from padding) will have length ≈ 0
        mask = (edge_lengths.squeeze(-1) > self.eps)  # [B, N]
        
        # Compute normals using (dy, -dx) convention to match MGIoU2DPlus
        normals = torch.stack((edges[..., 1], -edges[..., 0]), dim=-1).to(edges.dtype)
        
        # Normalize normals to prevent extreme projection values
        # Use safe division: only normalize valid edges, leave degenerate ones as-is (will be masked out)
        normals = normals / (edge_lengths + _EPS)
        
        return normals, mask
    
    @staticmethod
    def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {reduction!r}")

class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int, use_mgiou: bool = False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.mgiou_loss = MGIoURect(representation="corner", reduction="sum") if use_mgiou else None

    @staticmethod
    def xyxy_to_corners(boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert bounding boxes from (x1, y1, x2, y2) format to 4 corner points.
        
        Args:
            boxes: Tensor of shape (N, 4) in (x1, y1, x2, y2) format
            
        Returns:
            Tensor of shape (N, 4, 2) representing 4 corner points for each box
        """
        x1, y1, x2, y2 = boxes.unbind(-1)
        corners = torch.stack([
            torch.stack([x1, y1], dim=-1),  # top-left
            torch.stack([x2, y1], dim=-1),  # top-right
            torch.stack([x2, y2], dim=-1),  # bottom-right
            torch.stack([x1, y2], dim=-1),  # bottom-left
        ], dim=1)
        return corners

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes. Returns (loss_iou, loss_dfl, mgiou_loss)."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        if self.mgiou_loss:
            pred_corners = self.xyxy_to_corners(pred_bboxes[fg_mask])
            target_corners = self.xyxy_to_corners(target_bboxes[fg_mask].to(pred_bboxes.dtype))
            
            mgiou_loss = self.mgiou_loss(
                pred_corners,
                target_corners,
                weight=weight.to(pred_bboxes.dtype),
                avg_factor=target_scores_sum
            )
            loss_iou = mgiou_loss
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
            mgiou_loss = torch.tensor(0.0).to(pred_dist.device)

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, mgiou_loss


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int, use_mgiou: bool = False):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)
        self.mgiou_loss = MGIoURect(representation="rect", reduction="sum") if use_mgiou else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes. Returns (loss_iou, loss_dfl, mgiou_loss)."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        if self.mgiou_loss:
            mgiou_loss = self.mgiou_loss(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask].to(pred_bboxes.dtype),
                weight=weight.to(pred_bboxes.dtype),
                avg_factor=target_scores_sum
            )
            loss_iou = mgiou_loss
        else:
            iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
            mgiou_loss = torch.tensor(0.0).to(pred_dist.device)

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, mgiou_loss


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


class PolygonLoss(nn.Module):
    """
    Criterion class for computing polygon losses using MGIoU or L2 distance.
    
    This loss supports two modes:
    1. MGIoU mode (use_mgiou=True): Uses MGIoUPoly for geometry-aware loss
    2. L2 mode (use_mgiou=False): Uses normalized L2 distance with area weighting
    
    NaN Prevention (L2 Mode):
    - Area values are clamped to minimum 1e-6 to prevent division by very small numbers
    - Additional epsilon (1e-9) added to denominator for numerical stability
    - kpt_loss_factor uses epsilon to avoid division by zero when all keypoints masked
    
    NaN Prevention (MGIoU Mode):
    - Inherits all NaN prevention mechanisms from MGIoUPoly class
    - See MGIoUPoly documentation for detailed safety features
    
    Debugging:
    - Set ULTRALYTICS_DEBUG_NAN=1 to enable runtime NaN checks in both modes
    """

    def __init__(self, use_mgiou: bool = False) -> None:
        """Initialize the PolygonLoss class with optional MGIoU loss."""
        super().__init__()  # dummy sigma, not used
        self.mgiou_loss = MGIoUPoly(reduction="mean") if use_mgiou else None

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate polygon loss using MGIoU or fallback to L2 distance.
        
        Args:
            pred_kpts: Predicted polygon vertices, shape (N, num_vertices, 2 or 3)
            gt_kpts: Ground truth polygon vertices, shape (N, num_vertices, 2 or 3)
            kpt_mask: Mask indicating valid vertices, shape (N, num_vertices)
            area: Area of bounding boxes, shape (N, 1)
            
        Returns:
            Tuple of (total_loss, mgiou_loss) where mgiou_loss is 0 if not using MGIoU
            
        Note:
            Set environment variable ULTRALYTICS_DEBUG_NAN=1 to enable runtime NaN checks.
        """
        # Debug checks for inputs
        _check_nan_tensor(pred_kpts, "pred_kpts", "PolygonLoss.forward input")
        _check_nan_tensor(gt_kpts, "gt_kpts", "PolygonLoss.forward input")
        _check_nan_tensor(area, "area", "PolygonLoss.forward input")
        
        if self.mgiou_loss:
            pred_poly = pred_kpts[..., :2]
            gt_poly = gt_kpts[..., :2]
            
            weights = area.squeeze(-1)
            
            mgiou_losses = self.mgiou_loss(pred_poly, gt_poly, weight=weights)
            
            total_loss = mgiou_losses.mean()
            
            # Debug check for output
            _check_nan_tensor(total_loss, "total_loss (mgiou)", "PolygonLoss.forward output")
            
            return total_loss, total_loss
        else:
            d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
            kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
            
            # Clamp area to prevent division by very small numbers that can cause numerical instability
            area_safe = area.clamp(min=1e-6)
            e = d / (area_safe + 1e-9)
            
            l2_loss = (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
            mgiou_loss = torch.tensor(0.0).to(pred_kpts.device)
            
            # Debug check for output
            _check_nan_tensor(l2_loss, "l2_loss", "PolygonLoss.forward output")
            
            return l2_loss, mgiou_loss

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
        self.use_mgiou = use_mgiou

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
        """Calculate the total loss and detach it for detection."""
        loss = torch.zeros(4, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        if self.use_mgiou:
            loss[3] *= self.hyp.box
        
        return loss * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
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
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
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

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

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
    ) -> torch.Tensor:
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
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

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

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, use_mgiou=False):  # model must be de-paralleled
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
        loss = torch.zeros(6, device=self.device)
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4], loss[5] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl
        if self.use_mgiou:
            loss[5] *= self.hyp.box

        return loss * batch_size, loss.detach()

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



class v8PolygonLoss(v8DetectionLoss):

    def __init__(self, model, use_mgiou: bool = False):
        super().__init__(model, use_mgiou=use_mgiou)
        self.poly_shape = model.model[-1].poly_shape
        self.bce_poly = nn.BCEWithLogitsLoss()
        npoly = self.poly_shape[0]
        self.polygon_loss = PolygonLoss(use_mgiou=use_mgiou)
        self.use_mgiou = use_mgiou

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros(5, device=self.device)
        feats, pred_poly = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_poly = pred_poly.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_poly = self.poly_decode(anchor_points, pred_poly.view(batch_size, -1, *self.poly_shape))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[3], box_mgiou = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            polygons = batch["polygons"].to(self.device).float().clone()
            polygons[..., 0] *= imgsz[1]
            polygons[..., 1] *= imgsz[0]

            poly_main_loss = self.calculate_polygon_loss(
                fg_mask, target_gt_idx, polygons, batch_idx, stride_tensor, target_bboxes, pred_poly
            )
            
            loss[1] = poly_main_loss
            loss[4] = poly_main_loss if self.use_mgiou else box_mgiou

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.polygon
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        if self.use_mgiou:
            loss[4] *= self.hyp.polygon
        else:
            loss[4] *= self.hyp.box

        return loss * batch_size, loss.detach()

    @staticmethod
    def poly_decode(anchor_points: torch.Tensor, pred_poly: torch.Tensor) -> torch.Tensor:
        y = pred_poly.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_polygon_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        gt_poly: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_poly: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate polygon loss with safety checks to prevent NaN.
        
        Potential NaN sources handled:
        1. Division by stride_tensor (ensured non-zero by design)
        2. Area calculation from degenerate bboxes (clamped in PolygonLoss)
        3. MGIoU computation with degenerate polygons (handled by MGIoUPoly)
        """
        batch_idx = batch_idx.flatten()
        polys_loss = torch.zeros(1, device=self.device)

        for i in range(pred_poly.shape[0]):
            fg_mask_i = masks[i]
            if fg_mask_i.sum():
                target_gt_idx_i = target_gt_idx[i][fg_mask_i]
                gt_matching_bs = batch_idx[target_gt_idx_i].long()
                gt_poly_scaled = gt_poly[gt_matching_bs]
                
                # Scale by stride - stride_tensor should never be zero by design
                # (it comes from model.stride which is set during model initialization)
                gt_poly_scaled[..., 0] /= stride_tensor[i]
                gt_poly_scaled[..., 1] /= stride_tensor[i]
                
                # Compute area from target bboxes - will be clamped in PolygonLoss if too small
                area = xyxy2xywh(target_bboxes[i][fg_mask_i])[:, 2:].prod(1, keepdim=True)
                pred_poly_i = pred_poly[i][fg_mask_i]
                poly_mask = torch.full_like(gt_poly_scaled[..., 0], True)
                poly_loss, _ = self.polygon_loss(pred_poly_i, gt_poly_scaled, poly_mask, area)
                polys_loss += poly_loss

        return polys_loss

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
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, mgiou_loss
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
            loss[0], loss[2], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        if self.use_mgiou:
            loss[3] *= self.hyp.box

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, mgiou_loss)

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

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

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

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
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

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

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
