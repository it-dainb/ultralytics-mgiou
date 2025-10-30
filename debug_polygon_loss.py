"""
Debug version of Polygon Loss with extensive validation and logging.
This file contains instrumented versions to identify NaN causes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import warnings

_EPS = 1e-9
_SAFE_EPS = 1e-7  # Safer epsilon for testing


def check_tensor(tensor: torch.Tensor, name: str, step: str) -> dict:
    """Comprehensive tensor validation with detailed diagnostics."""
    info = {
        'name': name,
        'step': step,
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'min': tensor.min().item() if tensor.numel() > 0 else None,
        'max': tensor.max().item() if tensor.numel() > 0 else None,
        'mean': tensor.mean().item() if tensor.numel() > 0 else None,
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
    }
    
    if info['has_nan'] or info['has_inf']:
        print(f"⚠️  WARNING at {step}: {name}")
        print(f"   Shape: {info['shape']}, dtype: {info['dtype']}")
        print(f"   Has NaN: {info['has_nan']}, Has Inf: {info['has_inf']}")
        if info['min'] is not None:
            print(f"   Range: [{info['min']:.6e}, {info['max']:.6e}], Mean: {info['mean']:.6e}")
        
        # Count NaN/Inf occurrences
        if info['has_nan']:
            nan_count = torch.isnan(tensor).sum().item()
            print(f"   NaN count: {nan_count}/{tensor.numel()} ({100*nan_count/tensor.numel():.2f}%)")
        if info['has_inf']:
            inf_count = torch.isinf(tensor).sum().item()
            print(f"   Inf count: {inf_count}/{tensor.numel()} ({100*inf_count/tensor.numel():.2f}%)")
    
    return info


class MGIoUPolyDebug(nn.Module):
    """Debug version of MGIoUPoly with extensive validation."""
    
    def __init__(self, reduction="mean", loss_weight=1.0, eps=1e-6):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.debug_info = []

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: float | None = None,
    ) -> tuple[torch.Tensor, list]:
        """Compute MGIoU with full debugging."""
        self.debug_info = []
        self.debug_info.append(f"\n{'='*60}")
        self.debug_info.append(f"MGIoUPoly Forward Pass")
        self.debug_info.append(f"{'='*60}")
        
        check_tensor(pred, "pred", "input")
        check_tensor(target, "target", "input")
        
        B, N, _ = pred.shape
        M = target.shape[1]
        self.debug_info.append(f"Batch size: {B}, Pred vertices: {N}, Target vertices: {M}")
        
        red = reduction_override or self.reduction
        
        # Detect completely degenerate targets
        all_zero = (target.abs().sum(dim=(1, 2)) == 0)  # [B]
        self.debug_info.append(f"Degenerate targets: {all_zero.sum().item()}/{B}")
        
        losses = pred.new_zeros(B)
        
        # Fallback to L1 loss for degenerate targets
        if all_zero.any():
            pred_flat = pred[all_zero].view(all_zero.sum(), -1)
            target_flat = target[all_zero].view(all_zero.sum(), -1)
            l1 = F.l1_loss(pred_flat, target_flat, reduction="none")
            losses[all_zero] = l1.sum(dim=1).to(losses.dtype)
            check_tensor(losses[all_zero], "L1_losses", "degenerate_fallback")
        
        # Compute MGIoU for valid targets
        valid_mask = ~all_zero
        if valid_mask.any():
            self.debug_info.append(f"\nProcessing {valid_mask.sum().item()} valid polygons")
            
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
            
            # Check for extreme coordinates
            pred_info = check_tensor(pred_valid, "pred_valid", "before_sorting")
            target_info = check_tensor(target_valid, "target_valid", "before_sorting")
            
            # Sort vertices
            pred_sorted = self._sort_vertices(pred_valid)
            target_sorted = self._sort_vertices(target_valid)
            
            check_tensor(pred_sorted, "pred_sorted", "after_sorting")
            check_tensor(target_sorted, "target_sorted", "after_sorting")
            
            # Get axes and validity masks
            axes1, mask1 = self._axes_with_mask(pred_sorted)
            axes2, mask2 = self._axes_with_mask(target_sorted)
            
            check_tensor(axes1, "axes1", "after_axes_computation")
            check_tensor(axes2, "axes2", "after_axes_computation")
            self.debug_info.append(f"Valid axes - pred: {mask1.sum().item()}/{mask1.numel()}, "
                                 f"target: {mask2.sum().item()}/{mask2.numel()}")
            
            axes = torch.cat((axes1, axes2), dim=1)
            mask = torch.cat((mask1, mask2), dim=1)
            
            # Check how many samples have zero valid axes
            num_valid = mask.sum(dim=1)
            zero_valid_axes = (num_valid == 0).sum().item()
            if zero_valid_axes > 0:
                self.debug_info.append(f"⚠️  {zero_valid_axes} samples have NO valid axes!")
            
            # Project vertices
            proj1 = torch.bmm(pred_sorted.to(axes.dtype), axes.transpose(1, 2))
            proj2 = torch.bmm(target_sorted.to(axes.dtype), axes.transpose(1, 2))
            
            check_tensor(proj1, "proj1", "after_projection")
            check_tensor(proj2, "proj2", "after_projection")
            
            min1, _ = proj1.min(dim=1)
            max1, _ = proj1.max(dim=1)
            min2, _ = proj2.min(dim=1)
            max2, _ = proj2.max(dim=1)
            
            check_tensor(min1, "min1", "after_minmax")
            check_tensor(max1, "max1", "after_minmax")
            check_tensor(min2, "min2", "after_minmax")
            check_tensor(max2, "max2", "after_minmax")
            
            # Compute GIoU
            inter = (torch.minimum(max1, max2) - torch.maximum(min1, min2)).clamp(min=0.0)
            hull = torch.maximum(max1, max2) - torch.minimum(min1, min2)
            
            check_tensor(inter, "inter", "after_intersection")
            check_tensor(hull, "hull", "after_hull")
            
            # Check for zero or near-zero hull values
            small_hull = (hull < _SAFE_EPS).sum().item()
            if small_hull > 0:
                self.debug_info.append(f"⚠️  {small_hull} hull values < {_SAFE_EPS}")
            
            # Compute GIoU using union method
            union = (max1 - min1) + (max2 - min2) - inter
            check_tensor(union, "union", "after_union")
            
            # Check for zero or near-zero union values
            small_union = (union < _SAFE_EPS).sum().item()
            if small_union > 0:
                self.debug_info.append(f"⚠️  {small_union} union values < {_SAFE_EPS}")
            
            giou1d = inter / (union + _EPS) - (hull - union) / (hull + _EPS)
            
            check_tensor(giou1d, "giou1d", "after_giou_computation")
            
            # Masked mean
            giou1d_masked = giou1d * mask.to(giou1d.dtype)
            num_valid_clamped = num_valid.clamp(min=1)
            
            check_tensor(giou1d_masked, "giou1d_masked", "after_masking")
            check_tensor(num_valid_clamped.float(), "num_valid", "before_division")
            
            giou_val = giou1d_masked.sum(dim=1) / num_valid_clamped.squeeze()
            
            check_tensor(giou_val, "giou_val", "after_masked_mean")
            
            # Check GIoU range
            if giou_val.min() < -1.1 or giou_val.max() > 1.1:
                self.debug_info.append(f"⚠️  GIoU values out of expected range [-1, 1]: "
                                     f"[{giou_val.min().item():.6f}, {giou_val.max().item():.6f}]")
            
            loss_val = ((1.0 - giou_val) * 0.5).to(losses.dtype)
            check_tensor(loss_val, "loss_val", "after_loss_computation")
            
            losses[valid_mask] = loss_val
        
        check_tensor(losses, "losses", "before_weighting")
        
        # Weighting & reduction
        if weight is not None:
            weight = weight.view(-1) if weight.dim() > 1 else weight
            check_tensor(weight, "weight", "before_weighting")
            losses = losses * weight
            check_tensor(losses, "losses", "after_weighting")
            
            if avg_factor is None:
                avg_factor = weight.sum().clamp_min(1.0)
                self.debug_info.append(f"Auto avg_factor: {avg_factor.item():.6f}")
        
        loss = self._reduce(losses, red)
        check_tensor(loss, "loss", "after_reduction")
        
        if avg_factor is not None:
            self.debug_info.append(f"Dividing by avg_factor: {avg_factor}")
            check_tensor(torch.tensor(avg_factor), "avg_factor", "before_division")
            loss = loss / avg_factor
            check_tensor(loss, "loss", "after_avg_factor")
        
        final_loss = (loss * self.loss_weight).to(pred.dtype)
        check_tensor(final_loss, "final_loss", "final")
        
        return final_loss, self.debug_info

    @staticmethod
    def _sort_vertices(poly):
        """Sort vertices by angle from centroid."""
        B, N, _ = poly.shape
        center = poly.mean(dim=1, keepdim=True)
        angles = torch.atan2(poly[..., 1] - center[..., 1], poly[..., 0] - center[..., 0])
        indices = angles.argsort(dim=1)
        return torch.gather(poly, 1, indices.unsqueeze(-1).expand(-1, -1, 2))

    def _axes_with_mask(self, poly):
        """Compute edge normals and validity mask."""
        edges = poly.roll(-1, dims=1) - poly
        edge_lengths = torch.norm(edges, dim=-1)
        
        # Check edge statistics
        if edge_lengths.numel() > 0:
            min_edge = edge_lengths.min().item()
            max_edge = edge_lengths.max().item()
            zero_edges = (edge_lengths < self.eps).sum().item()
            self.debug_info.append(f"Edge lengths: [{min_edge:.6e}, {max_edge:.6e}], "
                                 f"Degenerate: {zero_edges}/{edge_lengths.numel()}")
        
        mask = edge_lengths > self.eps
        normals = torch.stack((edges[..., 1], -edges[..., 0]), dim=-1).to(edges.dtype)
        return normals, mask
    
    @staticmethod
    def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        if reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {reduction!r}")


class PolygonLossDebug(nn.Module):
    """Debug version of PolygonLoss."""
    
    def __init__(self, use_mgiou: bool = False) -> None:
        super().__init__()
        self.mgiou_loss = MGIoUPolyDebug(reduction="mean") if use_mgiou else None
        self.use_mgiou = use_mgiou
        self.debug_info = []

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Calculate polygon loss with debugging."""
        self.debug_info = []
        self.debug_info.append(f"\n{'='*60}")
        self.debug_info.append(f"PolygonLoss Forward Pass (use_mgiou={self.use_mgiou})")
        self.debug_info.append(f"{'='*60}")
        
        check_tensor(pred_kpts, "pred_kpts", "input")
        check_tensor(gt_kpts, "gt_kpts", "input")
        check_tensor(kpt_mask, "kpt_mask", "input")
        check_tensor(area, "area", "input")
        
        # Check for zero or very small areas
        small_area_count = (area < 1e-6).sum().item()
        if small_area_count > 0:
            self.debug_info.append(f"⚠️  {small_area_count} areas < 1e-6")
            self.debug_info.append(f"   Min area: {area.min().item():.6e}, Max area: {area.max().item():.6e}")
        
        if self.use_mgiou:
            pred_poly = pred_kpts[..., :2]
            gt_poly = gt_kpts[..., :2]
            
            weights = area.squeeze(-1)
            check_tensor(weights, "weights", "before_mgiou")
            
            mgiou_losses, debug_info = self.mgiou_loss(pred_poly, gt_poly, weight=weights)
            self.debug_info.extend(debug_info)
            
            total_loss = mgiou_losses.mean()
            check_tensor(total_loss, "total_loss", "final_mgiou")
            
            return total_loss, total_loss, self.debug_info
        else:
            d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
            check_tensor(d, "distance_squared", "after_distance")
            
            kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
            check_tensor(kpt_loss_factor, "kpt_loss_factor", "after_factor")
            
            # SAFE VERSION: Clamp area to prevent division by very small numbers
            area_safe = area.clamp(min=1e-6)
            e = d / (area_safe + 1e-9)
            check_tensor(e, "e", "after_normalization")
            
            l2_loss = (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
            check_tensor(l2_loss, "l2_loss", "final_l2")
            
            mgiou_loss = torch.tensor(0.0).to(pred_kpts.device)
            return l2_loss, mgiou_loss, self.debug_info


def test_mgiou_poly_debug():
    """Test MGIoUPoly with various edge cases."""
    print("\n" + "="*80)
    print("Testing MGIoUPoly Debug Version")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test 1: Normal polygons
    print("\n--- Test 1: Normal Polygons ---")
    mgiou = MGIoUPolyDebug(reduction="mean", eps=1e-6).to(device)
    pred = torch.tensor([
        [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ], device=device)
    target = torch.tensor([
        [[1.1, 1.1], [2.1, 1.1], [2.1, 2.1], [1.1, 2.1]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ], device=device)
    weights = torch.tensor([1.0, 1.0], device=device)
    
    loss, debug_info = mgiou(pred, target, weight=weights)
    print(f"Loss: {loss.item():.6f}")
    
    # Test 2: Degenerate polygon (all zeros)
    print("\n--- Test 2: Degenerate Polygon (all zeros) ---")
    pred_degen = torch.randn(2, 4, 2, device=device)
    target_degen = torch.zeros(2, 4, 2, device=device)
    weights_degen = torch.ones(2, device=device)
    
    loss_degen, debug_info = mgiou(pred_degen, target_degen, weight=weights_degen)
    print(f"Loss: {loss_degen.item():.6f}")
    
    # Test 3: Polygons with repeated vertices (padding)
    print("\n--- Test 3: Polygons with Repeated Vertices ---")
    pred_padded = torch.tensor([
        [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [2.0, 2.0]],  # Last vertex repeated
    ], device=device)
    target_padded = torch.tensor([
        [[1.1, 1.1], [2.1, 1.1], [2.1, 2.1], [2.1, 2.1]],
    ], device=device)
    weights_padded = torch.tensor([1.0], device=device)
    
    loss_padded, debug_info = mgiou(pred_padded, target_padded, weight=weights_padded)
    print(f"Loss: {loss_padded.item():.6f}")
    
    # Test 4: Extreme coordinates
    print("\n--- Test 4: Extreme Coordinates ---")
    pred_extreme = torch.tensor([
        [[1e6, 1e6], [2e6, 1e6], [2e6, 2e6], [1e6, 2e6]],
    ], device=device)
    target_extreme = torch.tensor([
        [[1e6, 1e6], [2e6, 1e6], [2e6, 2e6], [1e6, 2e6]],
    ], device=device)
    weights_extreme = torch.tensor([1.0], device=device)
    
    loss_extreme, debug_info = mgiou(pred_extreme, target_extreme, weight=weights_extreme)
    print(f"Loss: {loss_extreme.item():.6f}")
    
    # Test 5: Very small polygons
    print("\n--- Test 5: Very Small Polygons ---")
    pred_small = torch.tensor([
        [[0.0, 0.0], [1e-8, 0.0], [1e-8, 1e-8], [0.0, 1e-8]],
    ], device=device)
    target_small = torch.tensor([
        [[0.0, 0.0], [1e-8, 0.0], [1e-8, 1e-8], [0.0, 1e-8]],
    ], device=device)
    weights_small = torch.tensor([1.0], device=device)
    
    loss_small, debug_info = mgiou(pred_small, target_small, weight=weights_small)
    print(f"Loss: {loss_small.item():.6f}")
    
    print("\n" + "="*80)
    print("MGIoUPoly Debug Tests Complete")
    print("="*80)


def test_polygon_loss_debug():
    """Test PolygonLoss with various scenarios."""
    print("\n" + "="*80)
    print("Testing PolygonLoss Debug Version")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test with MGIoU
    print("\n--- Test 1: PolygonLoss with MGIoU ---")
    poly_loss_mgiou = PolygonLossDebug(use_mgiou=True).to(device)
    
    pred_kpts = torch.tensor([
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 2.0, 1.0]],
    ], device=device)
    gt_kpts = torch.tensor([
        [[1.1, 1.1, 1.0], [2.1, 1.1, 1.0], [2.1, 2.1, 1.0], [1.1, 2.1, 1.0]],
    ], device=device)
    kpt_mask = torch.ones(1, 4, device=device)
    area = torch.tensor([[1.0]], device=device)
    
    loss, mgiou_loss, debug_info = poly_loss_mgiou(pred_kpts, gt_kpts, kpt_mask, area)
    print(f"Total Loss: {loss.item():.6f}, MGIoU Loss: {mgiou_loss.item():.6f}")
    
    # Test with L2
    print("\n--- Test 2: PolygonLoss with L2 ---")
    poly_loss_l2 = PolygonLossDebug(use_mgiou=False).to(device)
    
    loss_l2, mgiou_loss_l2, debug_info = poly_loss_l2(pred_kpts, gt_kpts, kpt_mask, area)
    print(f"L2 Loss: {loss_l2.item():.6f}, MGIoU Loss: {mgiou_loss_l2.item():.6f}")
    
    # Test with very small area
    print("\n--- Test 3: Very Small Area ---")
    area_small = torch.tensor([[1e-10]], device=device)
    
    loss_small_area, _, debug_info = poly_loss_l2(pred_kpts, gt_kpts, kpt_mask, area_small)
    print(f"Loss with tiny area: {loss_small_area.item():.6f}")
    
    print("\n" + "="*80)
    print("PolygonLoss Debug Tests Complete")
    print("="*80)


if __name__ == "__main__":
    test_mgiou_poly_debug()
    test_polygon_loss_debug()
