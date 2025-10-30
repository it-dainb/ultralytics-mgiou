"""Test that Polygon head initialization doesn't cause NaN in classification logits."""

import torch
from ultralytics.nn.modules.head import Polygon

def test_polygon_head_bias_init():
    """Test that bias_init() properly initializes all heads after stride is set."""
    
    # Create a Polygon head
    nc = 1  # number of classes
    np_points = 8  # number of polygon points
    ch = (64, 128, 256)  # channel sizes
    
    head = Polygon(nc=nc, np=np_points, ch=ch)
    
    # Before bias_init, stride should be zeros
    assert torch.all(head.stride == 0), "Stride should be zeros before initialization"
    
    # Simulate what tasks.py does: set stride, then call bias_init()
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.bias_init()
    
    # Check that cv3 (classification head) biases are properly initialized
    print("\n=== Checking classification head (cv3) bias initialization ===")
    for i, (conv, stride) in enumerate(zip(head.cv3, head.stride)):
        final_conv = conv[-1]  # Last conv layer
        bias = final_conv.bias.data[:nc]
        expected = torch.tensor([torch.log(torch.tensor(5 / nc / (640 / stride) ** 2))])
        
        print(f"Layer {i} (stride={stride}):")
        print(f"  Expected bias: {expected.item():.6f}")
        print(f"  Actual bias:   {bias.item():.6f}")
        print(f"  Is finite:     {torch.isfinite(bias).all().item()}")
        
        assert torch.isfinite(bias).all(), f"Layer {i}: bias is not finite! Got {bias}"
        assert torch.allclose(bias, expected, rtol=1e-5), f"Layer {i}: bias mismatch"
    
    # Check that cv2 (box head) biases are set to 1.0
    print("\n=== Checking box head (cv2) bias initialization ===")
    for i, conv in enumerate(head.cv2):
        final_conv = conv[-1]
        bias = final_conv.bias.data
        print(f"Layer {i}: min={bias.min().item():.6f}, max={bias.max().item():.6f}, mean={bias.mean().item():.6f}")
        assert torch.allclose(bias, torch.ones_like(bias)), f"Layer {i}: cv2 bias should be 1.0"
    
    # Check that cv4 (polygon head) uses PyTorch default initialization (finite, small values)
    print("\n=== Checking polygon head (cv4) bias initialization ===")
    for i, conv in enumerate(head.cv4):
        final_conv = conv[-1]
        bias = final_conv.bias.data
        weight = final_conv.weight.data
        print(f"Layer {i}: bias_range=[{bias.min().item():.6f}, {bias.max().item():.6f}], weight_range=[{weight.min().item():.6f}, {weight.max().item():.6f}]")
        # PyTorch default initialization should produce finite, reasonably small values
        assert torch.isfinite(bias).all(), f"Layer {i}: cv4 bias contains NaN/Inf!"
        assert torch.isfinite(weight).all(), f"Layer {i}: cv4 weight contains NaN/Inf!"
        # Default initialization should be in reasonable range (typically |x| < 1 for small networks)
        assert bias.abs().max() < 10.0, f"Layer {i}: cv4 bias too large"
        assert weight.abs().max() < 10.0, f"Layer {i}: cv4 weight too large"
    
    print("\n✅ All bias initializations are correct!")
    
    # Now test a forward pass to ensure no NaN
    print("\n=== Testing forward pass ===")
    batch_size = 2
    x = [
        torch.randn(batch_size, ch[0], 64, 64),
        torch.randn(batch_size, ch[1], 32, 32),
        torch.randn(batch_size, ch[2], 16, 16),
    ]
    
    head.training = True
    feats, poly = head(x)
    
    # Check that outputs are finite
    for i, feat in enumerate(feats):
        print(f"Feature {i}: shape={feat.shape}, has_nan={torch.isnan(feat).any().item()}, has_inf={torch.isinf(feat).any().item()}")
        assert torch.isfinite(feat).all(), f"Feature {i} contains NaN/Inf!"
    
    print(f"Polygon output: shape={poly.shape}, has_nan={torch.isnan(poly).any().item()}, has_inf={torch.isinf(poly).any().item()}")
    assert torch.isfinite(poly).all(), "Polygon output contains NaN/Inf!"
    
    print("\n✅ Forward pass produces finite outputs!")

if __name__ == "__main__":
    test_polygon_head_bias_init()
    print("\n" + "="*60)
    print("SUCCESS: Polygon head initialization is correct!")
    print("="*60)
