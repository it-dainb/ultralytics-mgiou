"""
Comprehensive integration test for MGIoUPoly usage in PolygonLoss and metrics.
Tests gradient flow, mathematical correctness, and return types.
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import MGIoUPoly, PolygonLoss
from ultralytics.utils.metrics import poly_iou

print("="*80)
print("Comprehensive MGIoUPoly Integration Test")
print("="*80)

def test_mgiou_basic_interface():
    """Test 1: Basic MGIoUPoly interface and return types"""
    print("\n" + "="*80)
    print("Test 1: Basic MGIoUPoly Interface")
    print("="*80)
    
    torch.manual_seed(42)
    mgiou = MGIoUPoly(reduction="mean", loss_weight=1.0, eps=1e-6)
    
    # Test basic forward pass
    pred = torch.randn(4, 4, 2, requires_grad=True)
    target = torch.randn(4, 4, 2)
    
    loss = mgiou(pred, target)
    
    print(f"  Input shapes: pred={pred.shape}, target={target.shape}")
    print(f"  Output shape: {loss.shape}")
    print(f"  Output type: {loss.dtype}")
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Loss requires_grad: {loss.requires_grad}")
    
    # Check return type
    assert loss.shape == torch.Size([]), f"Expected scalar output with reduction='mean', got {loss.shape}"
    assert loss.requires_grad, "Loss should require gradient"
    assert torch.isfinite(loss), "Loss should be finite"
    
    print("  ✓ Basic interface correct")


def test_mgiou_gradient_flow():
    """Test 2: Gradient flow through MGIoUPoly"""
    print("\n" + "="*80)
    print("Test 2: Gradient Flow")
    print("="*80)
    
    torch.manual_seed(42)
    mgiou = MGIoUPoly(reduction="mean")
    
    pred = torch.randn(4, 4, 2, requires_grad=True)
    target = torch.randn(4, 4, 2)
    
    loss = mgiou(pred, target)
    loss.backward()
    
    assert pred.grad is not None, "Gradients should exist"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    assert not torch.isinf(pred.grad).any(), "Gradients should not contain Inf"
    assert (pred.grad != 0).any(), "Gradients should not be all zeros"
    
    print(f"  Gradient shape: {pred.grad.shape}")
    print(f"  Gradient mean: {pred.grad.mean().item():.6e}")
    print(f"  Gradient std: {pred.grad.std().item():.6e}")
    print(f"  Gradient min: {pred.grad.min().item():.6e}")
    print(f"  Gradient max: {pred.grad.max().item():.6e}")
    print(f"  Non-zero gradients: {(pred.grad != 0).sum().item()}/{pred.grad.numel()}")
    print("  ✓ Gradient flow correct")


def test_mgiou_with_weights():
    """Test 3: MGIoUPoly with per-sample weights"""
    print("\n" + "="*80)
    print("Test 3: Per-Sample Weights")
    print("="*80)
    
    torch.manual_seed(42)
    mgiou = MGIoUPoly(reduction="mean")
    
    pred = torch.randn(4, 4, 2, requires_grad=True)
    target = torch.randn(4, 4, 2)
    weights = torch.tensor([1.0, 0.5, 0.0, 2.0])
    
    # Test with weights
    loss_weighted = mgiou(pred, target, weight=weights)
    
    # Test without weights
    loss_unweighted = mgiou(pred, target)
    
    print(f"  Loss without weights: {loss_unweighted.item():.6f}")
    print(f"  Loss with weights: {loss_weighted.item():.6f}")
    print(f"  Weight values: {weights.tolist()}")
    
    assert loss_weighted.requires_grad, "Weighted loss should require gradient"
    assert torch.isfinite(loss_weighted), "Weighted loss should be finite"
    
    # Verify gradient flow with weights
    loss_weighted.backward()
    assert pred.grad is not None, "Gradients should exist with weights"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN with weights"
    
    print("  ✓ Weighted loss correct")


def test_mgiou_with_avg_factor():
    """Test 4: MGIoUPoly with avg_factor"""
    print("\n" + "="*80)
    print("Test 4: Average Factor")
    print("="*80)
    
    torch.manual_seed(42)
    mgiou = MGIoUPoly(reduction="mean")
    
    pred = torch.randn(4, 4, 2, requires_grad=True)
    target = torch.randn(4, 4, 2)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    avg_factor = 2.0
    
    loss = mgiou(pred, target, weight=weights, avg_factor=avg_factor)
    
    print(f"  Loss value: {loss.item():.6f}")
    print(f"  Avg factor: {avg_factor}")
    
    assert loss.requires_grad, "Loss should require gradient"
    assert torch.isfinite(loss), "Loss should be finite"
    
    loss.backward()
    assert pred.grad is not None, "Gradients should exist with avg_factor"
    
    print("  ✓ Average factor handling correct")


def test_mgiou_reduction_modes():
    """Test 5: MGIoUPoly different reduction modes"""
    print("\n" + "="*80)
    print("Test 5: Reduction Modes")
    print("="*80)
    
    torch.manual_seed(42)
    pred = torch.randn(4, 4, 2, requires_grad=True)
    target = torch.randn(4, 4, 2)
    
    # Test all reduction modes
    for reduction in ["none", "mean", "sum"]:
        mgiou = MGIoUPoly(reduction=reduction)
        loss = mgiou(pred, target)
        
        if reduction == "none":
            expected_shape = torch.Size([4])
            assert loss.shape == expected_shape, f"Expected shape {expected_shape}, got {loss.shape}"
        else:
            expected_shape = torch.Size([])
            assert loss.shape == expected_shape, f"Expected scalar, got {loss.shape}"
        
        print(f"  Reduction '{reduction}': shape={loss.shape}, value={loss.mean().item():.6f}")
    
    print("  ✓ All reduction modes correct")


def test_polygon_loss_mgiou_mode():
    """Test 6: PolygonLoss with MGIoU mode"""
    print("\n" + "="*80)
    print("Test 6: PolygonLoss with MGIoU Mode")
    print("="*80)
    
    torch.manual_seed(42)
    poly_loss = PolygonLoss(use_mgiou=True)
    
    # Simulate polygon loss inputs
    pred_kpts = torch.randn(4, 4, 2, requires_grad=True)
    gt_kpts = torch.randn(4, 4, 2)
    kpt_mask = torch.ones(4, 4)
    area = torch.rand(4, 1) * 100 + 10  # Areas between 10 and 110
    
    total_loss, mgiou_loss = poly_loss(pred_kpts, gt_kpts, kpt_mask, area)
    
    print(f"  Input shapes:")
    print(f"    pred_kpts: {pred_kpts.shape}")
    print(f"    gt_kpts: {gt_kpts.shape}")
    print(f"    area: {area.shape}")
    print(f"  Output:")
    print(f"    total_loss: {total_loss.item():.6f}")
    print(f"    mgiou_loss: {mgiou_loss.item():.6f}")
    print(f"    total_loss == mgiou_loss: {torch.allclose(total_loss, mgiou_loss)}")
    
    # Verify return types
    assert total_loss.shape == torch.Size([]), "total_loss should be scalar"
    assert mgiou_loss.shape == torch.Size([]), "mgiou_loss should be scalar"
    assert total_loss.requires_grad, "total_loss should require gradient"
    assert torch.isfinite(total_loss), "total_loss should be finite"
    assert torch.isfinite(mgiou_loss), "mgiou_loss should be finite"
    
    # In MGIoU mode, total_loss should equal mgiou_loss
    assert torch.allclose(total_loss, mgiou_loss), "In MGIoU mode, total_loss should equal mgiou_loss"
    
    # Test gradient flow
    total_loss.backward()
    assert pred_kpts.grad is not None, "Gradients should flow through PolygonLoss"
    assert not torch.isnan(pred_kpts.grad).any(), "Gradients should not contain NaN"
    
    print(f"  Gradient flow:")
    print(f"    Gradient exists: {pred_kpts.grad is not None}")
    print(f"    Gradient mean: {pred_kpts.grad.mean().item():.6e}")
    print(f"    Gradient std: {pred_kpts.grad.std().item():.6e}")
    print("  ✓ PolygonLoss MGIoU mode correct")


def test_polygon_loss_l2_mode():
    """Test 7: PolygonLoss with L2 mode"""
    print("\n" + "="*80)
    print("Test 7: PolygonLoss with L2 Mode")
    print("="*80)
    
    torch.manual_seed(42)
    poly_loss = PolygonLoss(use_mgiou=False)
    
    pred_kpts = torch.randn(4, 4, 2, requires_grad=True)
    gt_kpts = torch.randn(4, 4, 2)
    kpt_mask = torch.ones(4, 4)
    area = torch.rand(4, 1) * 100 + 10
    
    total_loss, mgiou_loss = poly_loss(pred_kpts, gt_kpts, kpt_mask, area)
    
    print(f"  Output:")
    print(f"    total_loss (L2): {total_loss.item():.6f}")
    print(f"    mgiou_loss: {mgiou_loss.item():.6f}")
    
    # In L2 mode, mgiou_loss should be 0
    assert mgiou_loss.item() == 0.0, "In L2 mode, mgiou_loss should be 0"
    assert total_loss.requires_grad, "total_loss should require gradient"
    assert torch.isfinite(total_loss), "total_loss should be finite"
    
    # Test gradient flow
    total_loss.backward()
    assert pred_kpts.grad is not None, "Gradients should flow in L2 mode"
    
    print("  ✓ PolygonLoss L2 mode correct")


def test_poly_iou_metric():
    """Test 8: poly_iou metric function"""
    print("\n" + "="*80)
    print("Test 8: poly_iou Metric Function")
    print("="*80)
    
    torch.manual_seed(42)
    
    # Create test polygons
    poly1 = torch.randn(3, 4, 2)
    poly2 = torch.randn(5, 4, 2)
    
    giou_matrix = poly_iou(poly1, poly2)
    
    print(f"  Input shapes:")
    print(f"    poly1: {poly1.shape}")
    print(f"    poly2: {poly2.shape}")
    print(f"  Output shape: {giou_matrix.shape}")
    print(f"  GIoU range: [{giou_matrix.min().item():.6f}, {giou_matrix.max().item():.6f}]")
    print(f"  GIoU mean: {giou_matrix.mean().item():.6f}")
    
    # Verify output shape and range
    assert giou_matrix.shape == torch.Size([3, 5]), f"Expected shape (3, 5), got {giou_matrix.shape}"
    assert giou_matrix.min() >= -1.0, "GIoU should be >= -1"
    assert giou_matrix.max() <= 1.0, "GIoU should be <= 1"
    assert not torch.isnan(giou_matrix).any(), "GIoU should not contain NaN"
    
    print("  ✓ poly_iou metric correct")


def test_edge_cases():
    """Test 9: Edge cases"""
    print("\n" + "="*80)
    print("Test 9: Edge Cases")
    print("="*80)
    
    torch.manual_seed(42)
    mgiou = MGIoUPoly(reduction="mean")
    
    # Test 9a: Degenerate target (all zeros)
    pred = torch.randn(2, 4, 2, requires_grad=True)
    target = torch.zeros(2, 4, 2)
    
    loss = mgiou(pred, target)
    print(f"  Degenerate target loss: {loss.item():.6f} (should use L1 fallback)")
    assert torch.isfinite(loss), "Loss should be finite with degenerate target"
    
    loss.backward()
    assert pred.grad is not None, "Gradients should exist with degenerate target"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaN"
    
    # Test 9b: Identical polygons
    pred = torch.randn(2, 4, 2, requires_grad=True)
    target = pred.detach().clone()
    
    loss = mgiou(pred, target)
    print(f"  Identical polygons loss: {loss.item():.6f} (should be close to 0)")
    assert torch.isfinite(loss), "Loss should be finite with identical polygons"
    
    # Test 9c: Very small polygons
    pred = torch.randn(2, 4, 2, requires_grad=True) * 1e-5
    target = torch.randn(2, 4, 2) * 1e-5
    
    loss = mgiou(pred, target)
    print(f"  Very small polygons loss: {loss.item():.6f}")
    assert torch.isfinite(loss), "Loss should be finite with very small polygons"
    
    # Test 9d: Very large polygons
    pred = torch.randn(2, 4, 2, requires_grad=True) * 1e5
    target = torch.randn(2, 4, 2) * 1e5
    
    loss = mgiou(pred, target)
    print(f"  Very large polygons loss: {loss.item():.6f}")
    assert torch.isfinite(loss), "Loss should be finite with very large polygons"
    
    print("  ✓ All edge cases handled correctly")


def test_mathematical_consistency():
    """Test 10: Mathematical consistency"""
    print("\n" + "="*80)
    print("Test 10: Mathematical Consistency")
    print("="*80)
    
    torch.manual_seed(42)
    
    # Test that loss decreases when predictions improve
    target = torch.randn(2, 4, 2)
    
    # Bad prediction (far from target)
    pred_bad = target + torch.randn(2, 4, 2) * 2.0
    pred_bad.requires_grad = True
    
    # Good prediction (close to target)
    pred_good = target + torch.randn(2, 4, 2) * 0.1
    pred_good.requires_grad = True
    
    mgiou = MGIoUPoly(reduction="mean")
    
    loss_bad = mgiou(pred_bad, target)
    loss_good = mgiou(pred_good, target)
    
    print(f"  Loss with bad prediction: {loss_bad.item():.6f}")
    print(f"  Loss with good prediction: {loss_good.item():.6f}")
    print(f"  Loss decreases when prediction improves: {loss_good < loss_bad}")
    
    # Loss should be lower for better predictions
    assert loss_good < loss_bad, "Loss should decrease when predictions improve"
    
    # Test loss symmetry (swapping pred and target should give similar loss for symmetric loss)
    pred = torch.randn(2, 4, 2, requires_grad=True)
    target = torch.randn(2, 4, 2)
    
    loss1 = mgiou(pred, target)
    loss2 = mgiou(target, pred.detach())
    
    print(f"  Loss(pred, target): {loss1.item():.6f}")
    print(f"  Loss(target, pred): {loss2.item():.6f}")
    print(f"  Difference: {abs(loss1.item() - loss2.item()):.6e}")
    
    print("  ✓ Mathematical consistency verified")


def run_all_tests():
    """Run all integration tests"""
    tests = [
        test_mgiou_basic_interface,
        test_mgiou_gradient_flow,
        test_mgiou_with_weights,
        test_mgiou_with_avg_factor,
        test_mgiou_reduction_modes,
        test_polygon_loss_mgiou_mode,
        test_polygon_loss_l2_mode,
        test_poly_iou_metric,
        test_edge_cases,
        test_mathematical_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
    else:
        print(f"\n❌ {failed} TESTS FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
