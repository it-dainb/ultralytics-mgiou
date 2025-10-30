"""
Test realistic training scenarios that might cause NaN in polygon loss.
"""

import torch
import torch.nn.functional as F
from debug_polygon_loss import MGIoUPolyDebug, PolygonLossDebug, check_tensor


def test_batch_with_mixed_quality():
    """Test batch with mixture of challenging polygon cases (realistic scenarios)."""
    print("\n" + "="*80)
    print("Test: Batch with Mixed Quality Polygons")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mgiou = MGIoUPolyDebug(reduction="mean", eps=1e-6).to(device)
    
    # Create a batch with realistic challenging cases:
    # 1. Good polygon - normal size
    # 2. Very small polygon (near-degenerate but all 4 corners distinct)
    # 3. Nearly collapsed polygon (very close vertices but not identical)
    # 4. Extreme aspect ratio polygon
    pred = torch.tensor([
        # Good polygon
        [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
        # Very small polygon (near machine precision)
        [[0.0, 0.0], [1e-7, 0.0], [1e-7, 1e-7], [0.0, 1e-7]],
        # Nearly collapsed but distinct vertices
        [[5.0, 5.0], [5.001, 5.0], [5.001, 5.001], [5.0, 5.001]],
        # Extreme aspect ratio
        [[0.0, 0.0], [100.0, 0.0], [100.0, 0.01], [0.0, 0.01]],
    ], device=device)
    
    target = torch.tensor([
        [[11.0, 11.0], [21.0, 11.0], [21.0, 21.0], [11.0, 21.0]],
        [[0.0, 0.0], [1e-7, 0.0], [1e-7, 1e-7], [0.0, 1e-7]],
        [[5.5, 5.5], [5.502, 5.5], [5.502, 5.502], [5.5, 5.502]],
        [[1.0, 0.0], [101.0, 0.0], [101.0, 0.01], [1.0, 0.01]],
    ], device=device)
    
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    
    loss, debug_info = mgiou(pred, target, weight=weights)
    print(f"\nFinal Loss: {loss.item():.6f}")
    print(f"Has NaN: {torch.isnan(loss).any().item()}")
    print(f"Has Inf: {torch.isinf(loss).any().item()}")
    
    return not (torch.isnan(loss).any() or torch.isinf(loss).any())


def test_extreme_stride_division():
    """Test division by stride (common source of NaN)."""
    print("\n" + "="*80)
    print("Test: Extreme Stride Division")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simulate polygon coordinates after scaling by image size
    polygons = torch.tensor([
        [[640.0, 480.0], [700.0, 480.0], [700.0, 540.0], [640.0, 540.0]],
    ], device=device, dtype=torch.float32)
    
    # Simulate different stride values
    strides = torch.tensor([8.0, 16.0, 32.0, 64.0, 1e-10], device=device)
    
    for i, stride in enumerate(strides):
        print(f"\n  Stride: {stride.item():.2e}")
        poly_scaled = polygons.clone()
        poly_scaled[..., 0] /= stride
        poly_scaled[..., 1] /= stride
        
        info = check_tensor(poly_scaled, f"poly_scaled_stride_{stride.item():.2e}", "after_stride_division")
        
        if info['has_nan'] or info['has_inf']:
            print(f"  ❌ FAILED: Generated NaN/Inf with stride {stride.item():.2e}")
            return False
    
    print("\n  ✓ All stride divisions passed")
    return True


def test_extreme_area_boxes():
    """Test polygons with extreme (very small or very large) but valid bounding box areas."""
    print("\n" + "="*80)
    print("Test: Extreme Area Boxes")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poly_loss = PolygonLossDebug(use_mgiou=False).to(device)
    
    pred_kpts = torch.tensor([
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 2.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.1, 0.1, 1.0], [0.0, 0.1, 1.0]],
    ], device=device)
    
    gt_kpts = pred_kpts.clone()
    kpt_mask = torch.ones(2, 4, device=device)
    
    # Test various extreme area values (but all valid, non-zero)
    areas = [
        torch.tensor([[1.0], [0.01]], device=device),
        torch.tensor([[1e-6], [1e-8]], device=device),
        torch.tensor([[1e10], [1e15]], device=device),  # Very large areas
    ]
    
    for i, area in enumerate(areas):
        print(f"\n  Test areas: {area.view(-1).tolist()}")
        loss, _, debug_info = poly_loss(pred_kpts, gt_kpts, kpt_mask, area)
        
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"  ❌ FAILED: Generated NaN/Inf with areas {area.view(-1).tolist()}")
            return False
        else:
            print(f"  ✓ Loss: {loss.item():.6f}")
    
    print("\n  ✓ All area tests passed")
    return True


def test_gradient_flow():
    """Test that gradients don't produce NaN."""
    print("\n" + "="*80)
    print("Test: Gradient Flow")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mgiou = MGIoUPolyDebug(reduction="mean", eps=1e-6).to(device)
    
    # Create polygons with requires_grad=True
    pred = torch.tensor([
        [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
        [[5.0, 5.0], [10.0, 5.0], [10.0, 10.0], [5.0, 10.0]],
    ], device=device, requires_grad=True)
    
    target = torch.tensor([
        [[11.0, 11.0], [21.0, 11.0], [21.0, 21.0], [11.0, 21.0]],
        [[5.5, 5.5], [10.5, 5.5], [10.5, 10.5], [5.5, 10.5]],
    ], device=device)
    
    weights = torch.tensor([1.0, 1.0], device=device)
    
    loss, debug_info = mgiou(pred, target, weight=weights)
    
    print(f"  Forward pass loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    if pred.grad is None:
        print("  ❌ FAILED: No gradients computed")
        return False
    
    grad_info = check_tensor(pred.grad, "pred.grad", "after_backward")
    
    if grad_info['has_nan'] or grad_info['has_inf']:
        print("  ❌ FAILED: Gradients contain NaN/Inf")
        return False
    
    print(f"  ✓ Gradient range: [{grad_info['min']:.6e}, {grad_info['max']:.6e}]")
    return True


def test_large_batch_with_varying_weights():
    """Test large batch with varying weights (simulating target_scores)."""
    print("\n" + "="*80)
    print("Test: Large Batch with Varying Weights")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mgiou = MGIoUPolyDebug(reduction="mean", eps=1e-6).to(device)
    
    batch_size = 64
    
    # Create random polygons
    pred = torch.randn(batch_size, 4, 2, device=device) * 10 + 50
    target = pred + torch.randn(batch_size, 4, 2, device=device) * 2
    
    # Varying weights (some very small, some zero)
    weights = torch.rand(batch_size, device=device)
    weights[::8] = 0.0  # Every 8th weight is zero
    weights[1::8] = 1e-6  # Very small weights
    
    avg_factor = weights.sum().clamp_min(1.0)
    
    print(f"  Batch size: {batch_size}")
    print(f"  Weights - Min: {weights.min().item():.6e}, Max: {weights.max().item():.6e}")
    print(f"  Avg factor: {avg_factor.item():.6f}")
    
    loss, debug_info = mgiou(pred, target, weight=weights, avg_factor=avg_factor)
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("  ❌ FAILED: Generated NaN/Inf")
        return False
    
    print(f"  ✓ Loss: {loss.item():.6f}")
    return True


def run_all_tests():
    """Run all realistic scenario tests."""
    print("\n" + "="*80)
    print("RUNNING ALL REALISTIC SCENARIO TESTS")
    print("="*80)
    
    tests = [
        ("Mixed Quality Polygons", test_batch_with_mixed_quality),
        ("Extreme Stride Division", test_extreme_stride_division),
        ("Extreme Area Boxes", test_extreme_area_boxes),
        ("Gradient Flow", test_gradient_flow),
        ("Large Batch with Varying Weights", test_large_batch_with_varying_weights),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            print(f"\n  ❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    for test_name, passed, error in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! No NaN detected in any scenario.")
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed.")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
