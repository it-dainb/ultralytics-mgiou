"""
MGIoU2DPlusHybrid - Improved MGIoU loss with adaptive convex normalization.

This is a standalone implementation that can be used to replace MGIoUPoly in ultralytics.
"""

import torch
from torch import nn, Tensor
from typing import Optional

_EPS = 1e-6
_MIN_EDGE = 1e-3  # minimum allowed edge length


class MGIoU2DPlusHybrid(nn.Module):
    """
    Hybrid MGIoU loss with adaptive convex normalization.
    
    Combines:
      1. True polygonal MGIoU (multi-axis projection overlap)
      2. Optional convexity penalty normalized by polygon scale
      3. Safe numerical clipping and tanh saturation
      4. Automatic, robust adaptation of convex normalization exponent (no external input)
    
    --------------------------------------------------------------------------
    Adaptive normalization rule
    --------------------------------------------------------------------------
    The convex regularization term is normalized by mean edge length **L̄** raised to
    an exponent **p**:
        convex_term ∝ convex_raw / (L̄ ** p)
    
    Ideally, p ≈ 2 for scale invariance (area ∝ edge²), but during early training
    lower p (<2) gives stronger gradients and better shape correction.
    
    This module adapts **p** based on both model progress (MGIoU EMA) and
    geometric consistency (polygon scale spread).
    
    --------------------------------------------------------------------------
    Adaptive components
    --------------------------------------------------------------------------
    1. **MGIoU-driven schedule (training progress):**
       p = p_min + (p_max - p_min) * sigmoid(k * (mgiou_ema - thresh))
       - mgiou_ema : Exponential moving average of current MGIoU
       - p_min     : Lower bound for early stage (e.g. 1.2)
       - p_max     : Upper bound for late stage (theoretical scale-invariant 2.0)
       - k         : Steepness of sigmoid transition (default 5.0)
       - thresh    : MGIoU value where midpoint occurs (default 0.2)
       - ema_momentum : EMA smoothing (default 0.9)
       - dp_max    : Max per-step change to avoid instability (default 0.05)
    
    2. **Scale-robust normalization (geometry-driven):**
       Uses percentile-based edge length ratio instead of max/min to resist outliers.
       scale_ratio = (p95 / p05)
       adaptive_pow_scale = p_max - (p_max - p_min) * clamp(log10(scale_ratio) / decades_to_full, 0, 1)
       - decades_to_full : number of log10 decades to fully decay from p_max→p_min (default 3.0)
         meaning a 1000× scale difference saturates to p_min.
       - cap_ratio       : upper bound for ratio (default 1e4)
    
    The final exponent used:
        convex_norm_pow = mean(p_from_mgiou, p_from_scale)
    
    --------------------------------------------------------------------------
    Args:
        convex_weight: float
            Weight for convexity penalty. 0 disables it.
        reduction: str
            'none' | 'mean' | 'sum'
        loss_clip_max: float or None
            Optional clip for numeric stability.
        safe_saturate: bool
            Apply tanh to avoid extreme MGIoU ratios.
        convex_normalization_pow: float
            Initial base exponent for normalization.
        adaptive_convex_pow: bool
            Enable automatic adaptive scheduling.
        # MGIoU-based schedule parameters
        p_min, p_max, k, thresh, ema_momentum, dp_max
        # Scale-based adaptation parameters
        decades_to_full: float
            Controls how many log10 decades correspond to full transition.
        cap_ratio: float
            Hard cap for scale ratio.
        q_low, q_high: float
            Percentiles for robust scale ratio (default 0.05, 0.95)
    """

    def __init__(
        self,
        convex_weight: float = 0.25,
        reduction: str = "mean",
        loss_clip_max: Optional[float] = 2.0,
        safe_saturate: bool = True,
        convex_normalization_pow: float = 1.5,
        adaptive_convex_pow: bool = True,
        # MGIoU-based adaptive parameters
        p_min: float = 1.2,
        p_max: float = 2.0,
        k: float = 5.0,
        thresh: float = 0.2,
        ema_momentum: float = 0.9,
        dp_max: float = 0.05,
        # scale-based adaptive parameters
        decades_to_full: float = 3.0,
        cap_ratio: float = 1e4,
        q_low: float = 0.05,
        q_high: float = 0.95,
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none','mean' or 'sum'")

        self.convex_weight = float(convex_weight)
        self.reduction = reduction
        self.loss_clip_max = None if loss_clip_max is None else float(loss_clip_max)
        self.safe_saturate = bool(safe_saturate)
        self.convex_norm_pow = float(convex_normalization_pow)
        self.adaptive_convex_pow = adaptive_convex_pow

        # adaptive parameters
        self.p_min = p_min
        self.p_max = p_max
        self.k = k
        self.thresh = thresh
        self.ema_momentum = ema_momentum
        self.dp_max = dp_max
        self.decades_to_full = decades_to_full
        self.cap_ratio = cap_ratio
        self.q_low = q_low
        self.q_high = q_high

        # persistent states
        self.register_buffer("_mgiou_ema", torch.tensor(0.0))
        self.register_buffer("_prev_pow", torch.tensor(convex_normalization_pow))

    # ---------------------------------------------------- #
    #                   MAIN FORWARD                       #
    # ---------------------------------------------------- #

    def forward(
        self,
        pred: Tensor,      # [B,N,2]
        target: Tensor,    # [B,N,2]
        visible_mask: Optional[Tensor] = None,
        return_dict: bool = False,
    ):
        if visible_mask is not None:
            mask = visible_mask.bool().squeeze()
            pred, target = pred[mask], target[mask]

        # -------------------------------
        # Robust normalization of predictions
        # -------------------------------
        with torch.no_grad():
            pred_center = pred.mean(dim=1, keepdim=True)
            pred_scale = self._mean_edge_length(pred).clamp_min(_MIN_EDGE).unsqueeze(-1).unsqueeze(-1)

        pred_norm = (pred - pred_center) / pred_scale

        target_center = target.mean(dim=1, keepdim=True)
        target_scale = self._mean_edge_length(target).clamp_min(_EPS).unsqueeze(-1).unsqueeze(-1)
        target_norm = (target - target_center) / target_scale

        mgiou, diag = self._mgiou_batch(pred_norm, target_norm)
        loss = (1.0 - mgiou) * 0.5

        # -------------------------------
        # Adaptive convex normalization
        # -------------------------------
        if self.adaptive_convex_pow:
            with torch.no_grad():
                m = mgiou.mean().detach()
                self._mgiou_ema.mul_(self.ema_momentum).add_((1 - self.ema_momentum) * m)

                alpha = torch.sigmoid(self.k * (self._mgiou_ema - self.thresh))
                p_from_mgiou = self.p_min + (self.p_max - self.p_min) * alpha

                edge_scale = self._mean_edge_length(target)
                if edge_scale.numel() >= 6:
                    low = torch.quantile(edge_scale, self.q_low)
                    high = torch.quantile(edge_scale, self.q_high)
                else:
                    low, high = edge_scale.min(), edge_scale.max()

                ratio = (high / (low + _EPS)).clamp(1.0, self.cap_ratio)
                decades = torch.log10(ratio)
                norm = (decades / self.decades_to_full).clamp(0.0, 1.0)
                p_from_scale = self.p_max - (self.p_max - self.p_min) * norm

                p = 0.5 * (p_from_mgiou + p_from_scale)
                p = p.clamp(self._prev_pow - self.dp_max, self._prev_pow + self.dp_max)
                self._prev_pow.copy_(p)
                self.convex_norm_pow = float(p)

            diag["adaptive_pow"] = float(self.convex_norm_pow)
            diag["mgiou_ema"] = float(self._mgiou_ema)

        # -------------------------------
        # Convexity term (safe)
        # -------------------------------
        if self.convex_weight != 0.0:
            convex_mask = self._is_convex(target)
            convex_raw = self._convexity_loss(pred_norm)  # normalized pred
            edge_scale = self._mean_edge_length(target)
            denom = (edge_scale ** self.convex_norm_pow).clamp_min(_MIN_EDGE)
            convex_term = (convex_raw / denom) * self.convex_weight * convex_mask
            loss = loss + convex_term
            diag["convex_term"] = convex_term.detach()
            diag["convex_pow_used"] = float(self.convex_norm_pow)

        if self.loss_clip_max is not None:
            loss = torch.clamp(loss, 0.0, self.loss_clip_max)

        out = self._reduce(loss, self.reduction)

        if return_dict:
            diag.update({"per_sample_loss": loss.detach(), "mgiou": mgiou.detach()})
            return out, diag

        return out

    # ---------------------------------------------------- #
    #                MGIoU CORE ROUTINE                    #
    # ---------------------------------------------------- #

    def _mgiou_batch(self, c1: Tensor, c2: Tensor):
        scale = self._mean_edge_length(c2).unsqueeze(-1).unsqueeze(-1).clamp_min(_EPS)
        c1n = (c1 - c1.mean(dim=1, keepdim=True)) / scale
        c2n = (c2 - c2.mean(dim=1, keepdim=True)) / scale

        axes = torch.cat(
            (self._candidate_axes_batch(c1n), self._candidate_axes_batch(c2n)), dim=1
        )

        proj1 = torch.bmm(c1n, axes.transpose(1, 2))
        proj2 = torch.bmm(c2n, axes.transpose(1, 2))

        min1, _ = proj1.min(dim=1)
        max1, _ = proj1.max(dim=1)
        min2, _ = proj2.min(dim=1)
        max2, _ = proj2.max(dim=1)

        numerator = torch.minimum(max1, max2) - torch.maximum(min1, min2)
        denominator = torch.maximum(max1, max2) - torch.minimum(min1, min2)
        denominator = denominator.clamp_min(_EPS)
        numerator = torch.clamp(numerator, -denominator, denominator)

        giou1d = numerator / denominator

        if self.safe_saturate:
            giou1d = torch.tanh(giou1d)

        mgiou = giou1d.mean(dim=1)

        diag = {
            "giou1d_mean": giou1d.mean().detach(),
            "giou1d_std": giou1d.std().detach() if giou1d.numel() > 1 else torch.tensor(0.0),
            "edge_scale_mean": self._mean_edge_length(c2).detach(),
        }

        return mgiou.clamp(-1.0, 1.0), diag

    # ---------------------------------------------------- #
    #                  CONVEX HELPERS                      #
    # ---------------------------------------------------- #

    @staticmethod
    def _candidate_axes_batch(corners: Tensor) -> Tensor:
        center = corners.mean(dim=1, keepdim=True)
        rel = corners - center
        angles = torch.atan2(rel[..., 1], rel[..., 0])
        idx = angles.argsort(dim=1)
        sorted_corners = torch.gather(corners, 1, idx.unsqueeze(-1).expand_as(corners))
        edges = sorted_corners.roll(-1, dims=1) - sorted_corners
        normals = torch.stack((edges[..., 1], -edges[..., 0]), dim=-1)
        return normals

    @staticmethod
    def _convexity_loss(polygons: Tensor) -> Tensor:
        v_prev = polygons.roll(1, 1)
        v_next = polygons.roll(-1, 1)
        e1 = v_prev - polygons
        e2 = v_next - polygons
        cross = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
        sign_ref = torch.where(
            cross[:, 0:1].abs() <= _EPS, torch.ones_like(cross[:, 0:1]), cross[:, 0:1]
        ).sign()
        penalty = torch.clamp(-sign_ref * cross, 0.0)
        return penalty.mean(dim=1)

    @staticmethod
    def _is_convex(polygons: Tensor, eps: float = 1e-6) -> Tensor:
        v_prev = polygons.roll(1, 1)
        v_next = polygons.roll(-1, 1)
        e1 = v_prev - polygons
        e2 = v_next - polygons
        cross = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
        sign_ref = torch.where(
            cross[:, 0:1].abs() <= eps, torch.ones_like(cross[:, 0:1]), cross[:, 0:1]
        ).sign()
        same_sign = (cross * sign_ref > -eps).all(dim=1)
        return same_sign.float()

    @staticmethod
    def _mean_edge_length(polygons: Tensor) -> Tensor:
        edges = polygons.roll(-1, dims=1) - polygons
        return edges.norm(dim=-1).mean(dim=1).clamp(min=_EPS)

    @staticmethod
    def _reduce(loss: Tensor, reduction: str) -> Tensor:
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unsupported reduction: {reduction!r}")
