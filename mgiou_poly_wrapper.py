"""
Wrapper for MGIoU2DPlusHybrid to match the old MGIoUPoly interface.

This allows the new implementation to be used as a drop-in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from mgiou2d_plus_hybrid import MGIoU2DPlusHybrid, _EPS


class MGIoUPolyNew(nn.Module):
    """
    Drop-in replacement for MGIoUPoly using MGIoU2DPlusHybrid internally.
    
    This wrapper maintains backward compatibility with the old interface while
    using the improved MGIoU2DPlusHybrid implementation under the hood.
    """
    
    def __init__(
        self,
        # Core parameters
        reduction="mean",
        loss_weight=1.0,
        eps=1e-6,
        # MGIoU2DPlusHybrid parameters
        convex_weight=0.25,
        loss_clip_max=2.0,
        safe_saturate=True,
        convex_normalization_pow=1.5,
        adaptive_convex_pow=True,
        # MGIoU-based adaptive parameters
        p_min=1.2,
        p_max=2.0,
        k=5.0,
        thresh=0.2,
        ema_momentum=0.9,
        dp_max=0.05,
        # scale-based adaptive parameters
        decades_to_full=3.0,
        cap_ratio=1e4,
        q_low=0.05,
        q_high=0.95,
    ):
        """
        Initialize MGIoUPolyNew with both legacy and new parameters.
        
        Core Parameters:
            reduction: Loss reduction method: 'none', 'mean', or 'sum' (default: 'mean')
            loss_weight: Global weight for the loss (default: 1.0)
            eps: Threshold for detecting degenerate edges (default: 1e-6)
        
        MGIoU2DPlusHybrid Parameters:
            convex_weight: Weight for convexity penalty. 0 disables it. (default: 0.25)
            loss_clip_max: Optional clip for numeric stability (default: 2.0)
            safe_saturate: Apply tanh to avoid extreme MGIoU ratios (default: True)
            convex_normalization_pow: Initial base exponent for normalization (default: 1.5)
            adaptive_convex_pow: Enable automatic adaptive scheduling (default: True)
            p_min, p_max: Bounds for adaptive exponent (default: 1.2, 2.0)
            k: Steepness of sigmoid transition (default: 5.0)
            thresh: MGIoU value where midpoint occurs (default: 0.2)
            ema_momentum: EMA smoothing (default: 0.9)
            dp_max: Max per-step change to avoid instability (default: 0.05)
            decades_to_full: Log10 decades for full transition (default: 3.0)
            cap_ratio: Upper bound for scale ratio (default: 1e4)
            q_low, q_high: Percentiles for robust scale ratio (default: 0.05, 0.95)
        """
        super().__init__()
        
        # Core parameters
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        
        # Create internal MGIoU2DPlusHybrid instance
        self.mgiou_hybrid = MGIoU2DPlusHybrid(
            convex_weight=convex_weight,
            reduction="none",  # We'll handle reduction ourselves
            loss_clip_max=loss_clip_max,
            safe_saturate=safe_saturate,
            convex_normalization_pow=convex_normalization_pow,
            adaptive_convex_pow=adaptive_convex_pow,
            p_min=p_min,
            p_max=p_max,
            k=k,
            thresh=thresh,
            ema_momentum=ema_momentum,
            dp_max=dp_max,
            decades_to_full=decades_to_full,
            cap_ratio=cap_ratio,
            q_low=q_low,
            q_high=q_high,
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction_override: Optional[str] = None,
        avg_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute MGIoU loss for polygon predictions.
        
        Args:
            pred: Predicted polygons [B, N, 2] where N is number of vertices
            target: Target polygons [B, N, 2] where N is number of vertices (same as pred)
            weight: Per-sample weights [B] or [B, 1] (optional)
            reduction_override: Override default reduction method (optional)
            avg_factor: Normalization factor for loss (optional, computed from weights if not provided)
            
        Returns:
            Computed loss value (scalar if reduction != 'none', else [B])
        """
        B = pred.shape[0]
        red = reduction_override or self.reduction
        
        # Detect completely degenerate targets (like MGIoURect strategy)
        # This handles cases where target is all zeros or invalid
        all_zero = (target.abs().sum(dim=(1, 2)) == 0)  # [B]
        losses = pred.new_zeros(B)
        
        # Fallback to L1 loss for degenerate targets
        if all_zero.any():
            # Flatten polygons for L1 loss
            pred_flat = pred[all_zero].view(int(all_zero.sum().item()), -1)
            target_flat = target[all_zero].view(int(all_zero.sum().item()), -1)
            l1 = F.l1_loss(pred_flat, target_flat, reduction="none")
            losses[all_zero] = l1.sum(dim=1).to(losses.dtype)
        
        # Compute MGIoU for valid targets
        valid_mask = ~all_zero
        if valid_mask.any():
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
            
            # Use the new MGIoU2DPlusHybrid implementation
            loss_valid = self.mgiou_hybrid(pred_valid, target_valid)
            losses[valid_mask] = loss_valid.to(losses.dtype)
        
        # --- weighting & reduction (matching MGIoURect behavior) ---
        if weight is not None:
            weight = weight.view(-1) if weight.dim() > 1 else weight
            losses = losses * weight
            
            if avg_factor is None:
                avg_factor = weight.sum().clamp_min(1.0)
        
        loss = self._reduce(losses, red)
        
        if avg_factor is not None:
            loss = loss / avg_factor
        
        final_loss = (loss * self.loss_weight).to(pred.dtype)
        
        return final_loss
    
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


def test_compatibility():
    """Test that the new implementation works with the old interface."""
    print("Testing MGIoUPolyNew compatibility...")
    
    # Create instances
    old_style = MGIoUPolyNew(reduction="mean", loss_weight=1.0)
    print("✓ Created with old-style parameters")
    
    # Test forward pass
    torch.manual_seed(42)
    pred = torch.randn(2, 4, 2, requires_grad=True)
    target = torch.randn(2, 4, 2)
    
    loss = old_style(pred, target)
    print(f"✓ Forward pass works: loss = {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    assert pred.grad is not None
    print(f"✓ Backward pass works: grad norm = {pred.grad.norm().item():.4f}")
    
    # Test with weights
    weight = torch.tensor([1.0, 0.5])
    loss_weighted = old_style(pred, target, weight=weight)
    print(f"✓ Weighted loss works: loss = {loss_weighted.item():.4f}")
    
    # Test reduction modes
    for red in ["none", "mean", "sum"]:
        loss_fn = MGIoUPolyNew(reduction=red)
        result = loss_fn(pred, target)
        print(f"✓ Reduction '{red}' works: shape = {result.shape}, value = {result if red != 'none' else result.mean():.4f}")
    
    print("\n✅ All compatibility tests passed!")


if __name__ == "__main__":
    test_compatibility()
