"""
Test script for Hybrid Loss implementation in YOLO Polygon training.

This script validates:
1. PolygonLoss modes (L2, MGIoU, Hybrid)
2. Alpha scheduling functions (cosine, linear, step)
3. Epoch passing mechanism
4. Gradient flow and normalization
5. Loss computation correctness
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the loss classes
from ultralytics.utils.loss import PolygonLoss, v8PolygonLoss


def test_alpha_scheduling():
    """Test alpha scheduling functions across different schedules."""
    print("\n" + "="*80)
    print("TEST 1: Alpha Scheduling Functions")
    print("="*80)
    
    total_epochs = 100
    alpha_start = 0.9
    alpha_end = 0.2
    
    schedules = ["cosine", "linear", "step"]
    results = {}
    
    for schedule in schedules:
        print(f"\n--- Testing {schedule.upper()} schedule ---")
        loss_fn = PolygonLoss(
            use_hybrid=True,
            alpha_schedule=schedule,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            total_epochs=total_epochs
        )
        
        alphas = []
        epochs = [0, 25, 50, 75, 99]
        
        for epoch in epochs:
            alpha = loss_fn.get_alpha(epoch)
            alphas.append(alpha)
            print(f"  Epoch {epoch:3d}: α = {alpha:.4f}")
        
        # Store full schedule for visualization
        full_alphas = [loss_fn.get_alpha(e) for e in range(total_epochs)]
        results[schedule] = full_alphas
        
        # Validate boundaries
        assert abs(alphas[0] - alpha_start) < 0.001, f"{schedule}: First alpha should be {alpha_start}"
        assert abs(alphas[-1] - alpha_end) < 0.001, f"{schedule}: Last alpha should be {alpha_end}"
        print(f"  ✓ {schedule} schedule: Boundaries correct!")
    
    # Visualize all schedules
    plt.figure(figsize=(10, 6))
    for schedule, alphas in results.items():
        plt.plot(range(total_epochs), alphas, label=schedule.capitalize(), linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Alpha (L2 weight)', fontsize=12)
    plt.title('Alpha Scheduling Comparison: L2 → MGIoU Transition', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('alpha_schedule_comparison.png', dpi=150)
    print(f"\n✓ Saved visualization to: alpha_schedule_comparison.png")
    
    return True


def test_polygon_loss_modes():
    """Test PolygonLoss in different modes: L2, MGIoU, Hybrid."""
    print("\n" + "="*80)
    print("TEST 2: PolygonLoss Modes (L2, MGIoU, Hybrid)")
    print("="*80)
    
    # Create synthetic data
    batch_size = 4
    n_points = 4  # 4 polygon vertices
    
    # Ground truth polygons (normalized coordinates)
    gt_kpts = torch.tensor([
        [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],  # Square
        [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],  # Larger square
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],  # Smaller square
        [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9], [0.5, 0.5]],  # Irregular
    ], dtype=torch.float32).reshape(batch_size, n_points, 2)
    
    # Predictions (slightly perturbed)
    pred_kpts = gt_kpts + torch.randn_like(gt_kpts) * 0.05
    
    # Mask and area
    kpt_mask = torch.ones(batch_size, n_points, dtype=torch.bool)
    area = torch.tensor([0.16, 0.36, 0.04, 0.32], dtype=torch.float32)
    
    modes = [
        ("L2", {"use_mgiou": False, "use_hybrid": False}),
        ("MGIoU", {"use_mgiou": True, "use_hybrid": False}),
        ("Hybrid (epoch 0)", {"use_hybrid": True, "alpha_start": 0.9, "alpha_end": 0.2}),
        ("Hybrid (epoch 50)", {"use_hybrid": True, "alpha_start": 0.9, "alpha_end": 0.2}),
        ("Hybrid (epoch 99)", {"use_hybrid": True, "alpha_start": 0.9, "alpha_end": 0.2}),
    ]
    
    results = {}
    
    for mode_name, config in modes:
        print(f"\n--- Testing {mode_name} mode ---")
        
        loss_fn = PolygonLoss(**config)
        
        # Determine epoch for hybrid modes
        epoch = None
        if "epoch 0" in mode_name:
            epoch = 0
        elif "epoch 50" in mode_name:
            epoch = 50
        elif "epoch 99" in mode_name:
            epoch = 99
        
        # Compute loss (returns tuple: (total_loss, component_loss))
        loss_tuple = loss_fn(pred_kpts, gt_kpts, kpt_mask, area, epoch=epoch)
        loss = loss_tuple[0]  # Get the total loss
        
        print(f"  Loss value: {loss.item():.6f}")
        
        # Check for NaN/Inf
        assert not torch.isnan(loss), f"{mode_name}: Loss is NaN!"
        assert not torch.isinf(loss), f"{mode_name}: Loss is Inf!"
        assert loss.item() >= 0, f"{mode_name}: Loss is negative!"
        
        results[mode_name] = loss.item()
        
        if "Hybrid" in mode_name:
            alpha = loss_fn.get_alpha(epoch)
            print(f"  Alpha (L2 weight): {alpha:.4f}")
            print(f"  L2 Loss EMA: {loss_fn.l2_loss_ema:.6f}")
            print(f"  MGIoU Loss EMA: {loss_fn.mgiou_loss_ema:.6f}")
        
        print(f"  ✓ {mode_name}: Valid loss computed!")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    mode_names = list(results.keys())
    loss_values = list(results.values())
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    bars = plt.bar(range(len(mode_names)), loss_values, color=colors, alpha=0.7)
    plt.xlabel('Loss Mode', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Loss Comparison Across Different Modes', fontsize=14)
    plt.xticks(range(len(mode_names)), mode_names, rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('loss_mode_comparison.png', dpi=150)
    print(f"\n✓ Saved visualization to: loss_mode_comparison.png")
    
    return True


def test_gradient_flow():
    """Test gradient flow through different loss modes."""
    print("\n" + "="*80)
    print("TEST 3: Gradient Flow and Backpropagation")
    print("="*80)
    
    # Create data with gradients enabled
    batch_size = 4
    n_points = 4
    
    gt_kpts = torch.tensor([
        [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
        [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
        [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9], [0.5, 0.5]],
    ], dtype=torch.float32).reshape(batch_size, n_points, 2)
    
    modes = [
        ("L2", {"use_mgiou": False, "use_hybrid": False}),
        ("MGIoU", {"use_mgiou": True, "use_hybrid": False}),
        ("Hybrid", {"use_hybrid": True, "alpha_start": 0.9, "alpha_end": 0.2}),
    ]
    
    for mode_name, config in modes:
        print(f"\n--- Testing {mode_name} gradient flow ---")
        
        # Create predictions that require gradients
        pred_kpts = (gt_kpts + torch.randn_like(gt_kpts) * 0.05).clone().requires_grad_(True)
        kpt_mask = torch.ones(batch_size, n_points, dtype=torch.bool)
        area = torch.tensor([0.16, 0.36, 0.04, 0.32], dtype=torch.float32)
        
        loss_fn = PolygonLoss(**config)
        
        # Forward pass (returns tuple)
        loss_tuple = loss_fn(pred_kpts, gt_kpts, kpt_mask, area, epoch=50)
        loss = loss_tuple[0]  # Get the total loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert pred_kpts.grad is not None, f"{mode_name}: No gradients computed!"
        assert not torch.isnan(pred_kpts.grad).any(), f"{mode_name}: NaN in gradients!"
        assert not torch.isinf(pred_kpts.grad).any(), f"{mode_name}: Inf in gradients!"
        
        grad_norm = pred_kpts.grad.norm().item()
        grad_mean = pred_kpts.grad.abs().mean().item()
        grad_max = pred_kpts.grad.abs().max().item()
        
        print(f"  Gradient norm: {grad_norm:.6f}")
        print(f"  Gradient mean: {grad_mean:.6f}")
        print(f"  Gradient max:  {grad_max:.6f}")
        print(f"  ✓ {mode_name}: Gradients are valid!")
    
    return True


def test_epoch_passing():
    """Test epoch passing and alpha value updates."""
    print("\n" + "="*80)
    print("TEST 4: Epoch Passing Mechanism")
    print("="*80)
    
    # Create loss function with hybrid mode
    loss_fn = PolygonLoss(
        use_hybrid=True,
        alpha_schedule="cosine",
        alpha_start=0.9,
        alpha_end=0.2,
        total_epochs=100
    )
    
    # Create dummy data
    batch_size = 2
    n_points = 4
    
    gt_kpts = torch.rand(batch_size, n_points, 2)
    pred_kpts = gt_kpts + torch.randn_like(gt_kpts) * 0.05
    kpt_mask = torch.ones(batch_size, n_points, dtype=torch.bool)
    area = torch.tensor([0.16, 0.36], dtype=torch.float32)
    
    # Test different epochs
    test_epochs = [0, 10, 25, 50, 75, 90, 99]
    
    print("\nTesting alpha values at different epochs:")
    for epoch in test_epochs:
        loss_tuple = loss_fn(pred_kpts, gt_kpts, kpt_mask, area, epoch=epoch)
        loss = loss_tuple[0]  # Get the total loss
        alpha = loss_fn.get_alpha(epoch)
        
        print(f"  Epoch {epoch:3d}: α = {alpha:.4f}, Loss = {loss.item():.6f}")
        
        # Verify alpha is in valid range
        assert 0 <= alpha <= 1, f"Alpha out of range at epoch {epoch}: {alpha}"
    
    print("\n✓ Epoch passing mechanism works correctly!")
    return True


def test_v8_polygon_loss_integration():
    """Test v8PolygonLoss integration with hybrid loss."""
    print("\n" + "="*80)
    print("TEST 5: v8PolygonLoss Integration")
    print("="*80)
    
    # This test would require a full model setup, which is complex
    # Instead, we'll test that v8PolygonLoss accepts the hybrid parameters
    
    print("\nTesting v8PolygonLoss initialization with hybrid parameters...")
    
    # Create a minimal mock model with all required attributes
    class MockDetectHead:
        def __init__(self):
            self.poly_shape = (4, 2)  # 4 polygon vertices, (x,y) coordinates
            self.reg_max = 16
            self.nc = 80  # COCO classes
            self.stride = torch.tensor([8, 16, 32])
    
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.args = type('Args', (), {
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'kpt': 1.0,
                'pose': 12.0,
            })()
            self.nc = 80  # COCO classes
            self.device = torch.device('cpu')
            # Add a dummy parameter so parameters() works
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            # Add model attribute with detect head
            self.model = [None] * 10  # Dummy layers
            self.model.append(MockDetectHead())  # Detection head at the end
    
    mock_model = MockModel()
    
    # Test initialization with hybrid parameters
    try:
        v8_loss = v8PolygonLoss(
            model=mock_model,
            use_hybrid=True,
            alpha_schedule="cosine",
            alpha_start=0.9,
            alpha_end=0.2,
            total_epochs=100
        )
        
        print("  ✓ v8PolygonLoss initialized with hybrid parameters")
        
        # Test set_epoch method
        v8_loss.set_epoch(50)
        print("  ✓ set_epoch() method works")
        
        # Check that parameters were passed to underlying PolygonLoss
        assert hasattr(v8_loss, 'polygon_loss'), "No polygon_loss attribute"
        assert v8_loss.polygon_loss.use_hybrid, "use_hybrid not set"
        assert v8_loss.polygon_loss.alpha_schedule == "cosine", "alpha_schedule not set"
        
        print("  ✓ Parameters correctly passed to PolygonLoss")
        print("\n✓ v8PolygonLoss integration test passed!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error during v8PolygonLoss initialization: {e}")
        return False


def test_gradient_normalization():
    """Test gradient normalization via EMA tracking."""
    print("\n" + "="*80)
    print("TEST 6: Gradient Normalization (EMA Tracking)")
    print("="*80)
    
    loss_fn = PolygonLoss(
        use_hybrid=True,
        alpha_schedule="cosine",
        alpha_start=0.9,
        alpha_end=0.2,
        total_epochs=100
    )
    
    # Create data
    batch_size = 4
    n_points = 4
    gt_kpts = torch.rand(batch_size, n_points, 2)
    kpt_mask = torch.ones(batch_size, n_points, dtype=torch.bool)
    area = torch.rand(batch_size)
    
    print("\nTracking EMA values across multiple forward passes:")
    
    l2_emas = []
    mgiou_emas = []
    
    for i in range(10):
        # Different predictions each time
        pred_kpts = gt_kpts + torch.randn_like(gt_kpts) * 0.1
        
        loss_tuple = loss_fn(pred_kpts, gt_kpts, kpt_mask, area, epoch=i*10)
        # loss is already unpacked, we just need to trigger the forward pass
        
        l2_emas.append(loss_fn.l2_loss_ema)
        mgiou_emas.append(loss_fn.mgiou_loss_ema)
        
        if i % 3 == 0:
            print(f"  Iteration {i+1}: L2 EMA = {loss_fn.l2_loss_ema:.6f}, "
                  f"MGIoU EMA = {loss_fn.mgiou_loss_ema:.6f}")
    
    # Verify EMA values are updating
    assert l2_emas[-1] != l2_emas[0], "L2 EMA not updating"
    assert mgiou_emas[-1] != mgiou_emas[0], "MGIoU EMA not updating"
    
    # Verify EMA values are positive
    assert all(ema > 0 for ema in l2_emas), "L2 EMA has non-positive values"
    assert all(ema > 0 for ema in mgiou_emas), "MGIoU EMA has non-positive values"
    
    print("\n✓ Gradient normalization (EMA tracking) works correctly!")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print(" HYBRID LOSS IMPLEMENTATION TEST SUITE")
    print("="*80)
    
    tests = [
        ("Alpha Scheduling", test_alpha_scheduling),
        ("PolygonLoss Modes", test_polygon_loss_modes),
        ("Gradient Flow", test_gradient_flow),
        ("Epoch Passing", test_epoch_passing),
        ("v8PolygonLoss Integration", test_v8_polygon_loss_integration),
        ("Gradient Normalization", test_gradient_normalization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = "FAILED"
    
    # Print summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name:40s} {result}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Hybrid loss implementation is working correctly.")
        print("\nNext steps:")
        print("1. Try training with: yolo train model=yolo11n-polygon.yaml data=your_data.yaml use_hybrid=True")
        print("2. Compare results with L2-only and two-stage training")
        print("3. Monitor alpha values and loss magnitudes during training")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
