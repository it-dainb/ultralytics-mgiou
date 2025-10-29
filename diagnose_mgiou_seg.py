"""Diagnostic script to analyze MGIoU segmentation loss instability."""

import torch
import torch.nn.functional as F
import cv2
import numpy as np

def mask_to_polygon_corners_original(mask: torch.Tensor, epsilon_factor: float = 0.02):
    """Original implementation from loss.py"""
    try:
        mask_np = (mask.detach().cpu().numpy() > 0.5).astype('uint8')
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            return None
        
        arc_len = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * arc_len
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            approx = cv2.convexHull(contour)
            if len(approx) < 3:
                return None
        
        corners = torch.from_numpy(approx.reshape(-1, 2)).float().to(mask.device)

        if corners.shape[0] > 1:
            try:
                uniq_idx = np.unique(corners.cpu().numpy(), axis=0, return_index=True)[1]
                uniq_idx.sort()
                corners = corners[uniq_idx]
            except Exception:
                pass

        if corners.shape[0] < 3:
            return None

        centroid = corners.mean(dim=0)
        angles = torch.atan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        order = torch.argsort(angles)
        corners = corners[order]

        start_idx = torch.argmin(corners[:, 0] + corners[:, 1])
        corners = torch.roll(corners, -int(start_idx), dims=0)

        h, w = mask_np.shape
        corners[:, 0] = corners[:, 0].clamp(0, w - 1)
        corners[:, 1] = corners[:, 1].clamp(0, h - 1)
        corners = corners.round()

        return corners
    except Exception:
        return None


def create_test_masks():
    """Create realistic test masks to analyze corner count variability."""
    # 160x160 masks (typical segmentation mask size)
    size = 160
    
    # Create similar elliptical masks with slight variations
    masks = []
    for i in range(5):
        mask = torch.zeros(size, size)
        # Draw ellipse with slight variation
        center = (size // 2 + np.random.randint(-5, 5), size // 2 + np.random.randint(-5, 5))
        axes = (40 + np.random.randint(-5, 5), 30 + np.random.randint(-5, 5))
        angle = np.random.randint(0, 180)
        
        mask_np = mask.numpy()
        cv2.ellipse(mask_np, center, axes, angle, 0, 360, 1, -1)
        masks.append(torch.from_numpy(mask_np).float())
    
    return masks


def analyze_corner_count_stability():
    """Test how corner counts vary for similar masks."""
    print("=" * 60)
    print("CORNER COUNT STABILITY ANALYSIS")
    print("=" * 60)
    
    masks = create_test_masks()
    
    for epsilon in [0.01, 0.02, 0.03, 0.05]:
        print(f"\nEpsilon factor: {epsilon}")
        corner_counts = []
        
        for i, mask in enumerate(masks):
            corners = mask_to_polygon_corners_original(mask, epsilon_factor=epsilon)
            count = corners.shape[0] if corners is not None else 0
            corner_counts.append(count)
            print(f"  Mask {i}: {count} corners")
        
        if corner_counts:
            print(f"  Range: {min(corner_counts)} - {max(corner_counts)} corners")
            print(f"  Std dev: {np.std(corner_counts):.2f}")


def analyze_padding_effect():
    """Test how padding affects MGIoU calculation."""
    print("\n" + "=" * 60)
    print("PADDING EFFECT ANALYSIS")
    print("=" * 60)
    
    # Create a simple square polygon
    square = torch.tensor([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=torch.float32)
    
    # Pad it to different sizes
    for target_size in [4, 6, 8, 10]:
        if target_size > 4:
            pad = square[-1:].repeat(target_size - 4, 1)
            padded = torch.cat([square, pad], dim=0)
        else:
            padded = square
        
        print(f"\nPadded to {target_size} corners:")
        print(f"  Shape: {padded.shape}")
        print(f"  Unique points: {torch.unique(padded, dim=0).shape[0]}")
        print(f"  Last 3 points: {padded[-3:].tolist()}")


def analyze_coordinate_scale():
    """Test MGIoU sensitivity to coordinate scale."""
    print("\n" + "=" * 60)
    print("COORDINATE SCALE ANALYSIS")
    print("=" * 60)
    
    # Same polygon at different scales
    base = torch.tensor([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=torch.float32)
    
    for scale in [1.0, 2.0, 5.0, 10.0]:
        scaled = base * scale
        center = scaled.mean(dim=0)
        extents = (scaled.max(dim=0)[0] - scaled.min(dim=0)[0]).tolist()
        print(f"\nScale {scale}x:")
        print(f"  Center: {center.tolist()}")
        print(f"  Extents: {extents}")


def test_threshold_sensitivity():
    """Test how threshold affects corner extraction."""
    print("\n" + "=" * 60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Create a soft mask (like after sigmoid)
    size = 80
    mask = torch.zeros(size, size)
    center = (size // 2, size // 2)
    
    # Create gradient mask
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    dist = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = torch.sigmoid((20 - dist) * 0.5)  # Soft circular mask
    
    for threshold in [0.3, 0.5, 0.7]:
        binary_mask = (mask > threshold).float()
        corners = mask_to_polygon_corners_original(binary_mask, epsilon_factor=0.02)
        count = corners.shape[0] if corners is not None else 0
        print(f"\nThreshold {threshold}: {count} corners")
        print(f"  Pixels above threshold: {(binary_mask > 0).sum().item()}")


def proposed_fixes():
    """Display proposed fixes."""
    print("\n" + "=" * 60)
    print("PROPOSED FIXES")
    print("=" * 60)
    print("""
1. **Normalize coordinates to [0, 1]** before MGIoU calculation:
   - Divide by mask dimensions (mask_h, mask_w)
   - This makes MGIoU scale-invariant
   
2. **Use fixed corner count** (e.g., 8 or 12) via target-based sampling:
   - Instead of variable epsilon, sample fixed number of points
   - Use convex hull, then resample to N points uniformly
   
3. **Better padding strategy**:
   - Interpolate between corners instead of repeating last
   - Or use convex hull as fallback for consistent shape
   
4. **Adjust threshold**:
   - Use 0.3 or 0.4 instead of 0.5 for softer masks
   - Or use weighted contour based on probability
   
5. **Add smoothing/filtering**:
   - Apply Gaussian blur before contour extraction
   - Reduces noise-induced corner variation
   
6. **Normalize MGIoU loss separately**:
   - Don't mix with seg_loss scale (which uses BCE)
   - Use separate loss weight or adaptive scaling
   
7. **Use L1/L2 distance between polygon vertices as auxiliary**:
   - Penalize coordinate differences directly
   - More stable than MGIoU for training signal
""")


if __name__ == "__main__":
    analyze_corner_count_stability()
    analyze_padding_effect()
    analyze_coordinate_scale()
    test_threshold_sensitivity()
    proposed_fixes()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
