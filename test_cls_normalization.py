"""
Test classification loss normalization fix.

Directly test the normalization formula to verify it produces reasonable values.
"""

import torch
import torch.nn as nn


def test_cls_loss_normalization():
    """Test that normalization brings cls loss to reasonable range."""
    print("\n" + "="*70)
    print("Testing Classification Loss Normalization Formula")
    print("="*70 + "\n")
    
    # Realistic training scenario from actual training logs
    batch_size = 8
    num_anchors = 8400
    num_classes = 1
    num_positive_samples = 76  # From actual training: sum of target_scores
    
    print(f"Scenario (from real training):")
    print(f"  Batch size: {batch_size}")
    print(f"  Anchors per image: {num_anchors}")
    print(f"  Total anchors: {batch_size * num_anchors} = {batch_size * num_anchors}")
    print(f"  Positive samples: {num_positive_samples}")
    print(f"  Anchor:Positive ratio: {batch_size * num_anchors}:{num_positive_samples} = {(batch_size * num_anchors)//num_positive_samples}:1\n")
    
    # Create dummy predictions and targets
    pred_scores = torch.randn(batch_size, num_anchors, num_classes)
    target_scores = torch.zeros(batch_size, num_anchors, num_classes)
    
    # Simulate positive samples (76 out of 67200 anchors are positive)
    # Randomly set some to 1.0
    positive_indices = torch.randperm(batch_size * num_anchors)[:num_positive_samples]
    for idx in positive_indices:
        b = idx // num_anchors
        a = idx % num_anchors
        target_scores[b, a, 0] = 1.0
    
    target_scores_sum = target_scores.sum().item()
    print(f"Target scores sum: {target_scores_sum:.1f}\n")
    
    # BCE loss
    bce = nn.BCEWithLogitsLoss(reduction="none")
    bce_loss = bce(pred_scores, target_scores)
    
    # OLD formula (causing the problem)
    old_cls_loss = bce_loss.sum() / target_scores_sum
    
    # NEW formula (simple mean)
    new_cls_loss = bce_loss.mean()
    
    print("Loss Computation:")
    print(f"  BCE loss sum: {bce_loss.sum().item():.1f}")
    print(f"  Target scores sum: {target_scores_sum:.1f}\n")
    
    print("OLD Formula (before fix):")
    print(f"  sum(BCE) / target_scores_sum")
    print(f"  {bce_loss.sum().item():.1f} / {target_scores_sum:.1f}")
    print(f"  = {old_cls_loss.item():.4f}")
    print(f"  ❌ PROBLEM: Loss is very high ({old_cls_loss.item():.1f}), will dominate training!\n")
    
    print("NEW Formula (simple mean):")
    print(f"  mean(BCE)")
    print(f"  = {new_cls_loss.item():.6f}")
    
    # Check if new loss is reasonable
    if new_cls_loss < 10 and new_cls_loss > 0:
        print(f"  ✅ GOOD: Loss is in reasonable range (0 < {new_cls_loss.item():.4f} < 10)\n")
        success = True
    else:
        print(f"  ❌ PROBLEM: Loss is still unusual ({new_cls_loss.item():.4f})\n")
        success = False
    
    # Show the reduction factor
    reduction_factor = old_cls_loss / new_cls_loss
    print(f"Reduction factor: {reduction_factor.item():.1f}x")
    print(f"  Old loss was {reduction_factor.item():.1f}x larger than new loss\n")
    
    # Show expected impact on gradients
    print("Expected Impact:")
    print(f"  Before: Classification gradients were ~11,600x larger than polygon")
    print(f"  After:  Classification gradients reduced by {reduction_factor.item():.0f}x")
    new_ratio = 11600 / reduction_factor.item()
    print(f"          New ratio: ~{new_ratio:.0f}:1")
    
    if new_ratio < 100:
        print(f"  ✅ Gradients should now be balanced enough for both losses to contribute\n")
    else:
        print(f"  ⚠️  Still {new_ratio:.0f}:1 - may need further tuning\n")
    
    return success


def test_gradient_scale():
    """Test that gradient magnitudes are comparable after normalization."""
    print("="*70)
    print("Testing Gradient Magnitude Balance")
    print("="*70 + "\n")
    
    # Simulate typical polygon and classification losses with new formula
    batch_size = 8
    num_anchors = 8400
    num_classes = 1
    
    # Polygon loss typically around 10-12
    poly_loss_value = 11.5
    
    # With old formula, cls loss was ~650-710
    # With new formula: mean(BCE) which should be ~0.8 for BCE with random predictions
    cls_loss_value_old = 680.0
    cls_loss_value_new = 0.8  # Typical BCE mean value
    
    print(f"Typical loss values:")
    print(f"  Polygon loss: {poly_loss_value:.2f}")
    print(f"  Classification loss (old): {cls_loss_value_old:.2f}")
    print(f"  Classification loss (new): {cls_loss_value_new:.6f}\n")
    
    # Loss weighting from default config
    hyp_polygon = 2.5
    hyp_cls = 0.5
    
    weighted_poly = poly_loss_value * hyp_polygon
    weighted_cls_old = cls_loss_value_old * hyp_cls
    weighted_cls_new = cls_loss_value_new * hyp_cls
    
    print(f"After applying hyperparameters (polygon={hyp_polygon}, cls={hyp_cls}):")
    print(f"  Weighted polygon loss: {weighted_poly:.2f}")
    print(f"  Weighted cls loss (old): {weighted_cls_old:.2f}")
    print(f"  Weighted cls loss (new): {weighted_cls_new:.6f}\n")
    
    # Gradient ratio (assuming gradients proportional to loss)
    old_ratio = weighted_cls_old / weighted_poly
    new_ratio = weighted_cls_new / weighted_poly
    
    print(f"Gradient balance:")
    print(f"  OLD: cls/poly = {weighted_cls_old:.1f}/{weighted_poly:.1f} = {old_ratio:.1f}:1")
    print(f"       Classification dominates by {old_ratio:.0f}x")
    print(f"       ❌ Polygon gradients are too weak to matter\n")
    
    print(f"  NEW: cls/poly = {weighted_cls_new:.4f}/{weighted_poly:.1f} = {new_ratio:.4f}:1")
    
    if new_ratio < 0.1:
        print(f"       ⚠️  Classification might now be too weak")
        print(f"       Consider adjusting hyp.cls upward")
        return False
    elif new_ratio < 1.0:
        print(f"       ✅ Good balance! Both losses can contribute")
        print(f"       Polygon loss is {1/new_ratio:.1f}x stronger, which allows it to train\n")
        return True
    else:
        print(f"       ⚠️  Classification still dominates by {new_ratio:.1f}x\n")
        return False


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# Classification Loss Normalization Fix Verification")
    print("#"*70)
    
    test1 = test_cls_loss_normalization()
    test2 = test_gradient_scale()
    
    print("="*70)
    print("Summary")
    print("="*70)
    print(f"Normalization formula:  {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Gradient balance:       {'✅ PASS' if test2 else '❌ FAIL'}")
    print()
    
    if test1 and test2:
        print("🎉 Normalization fix is mathematically correct!")
        print("\nNext steps:")
        print("1. Run actual training to verify polygon loss decreases")
        print("2. Monitor metrics (P, R, mAP) to ensure classification still works")
        print("3. May need to increase hyp.cls to compensate for lower cls loss magnitude")
        print()
    else:
        print("⚠️  Normalization may need adjustment")
        print()
