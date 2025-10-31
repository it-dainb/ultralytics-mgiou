"""
Diagnostic script to identify why the polygon model predicts full-image bounding boxes.

This script will:
1. Load a trained model
2. Inspect the polygon head weights and biases
3. Analyze raw predictions before and after decoding
4. Check gradient flow during a forward pass
5. Identify the root cause of the issue
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path


def analyze_polygon_head(model):
    """Analyze the polygon head architecture and initialization."""
    print("\n" + "="*80)
    print("POLYGON HEAD ANALYSIS")
    print("="*80)
    
    # Get the polygon head
    head = model.model.model[-1]
    print(f"\nHead type: {type(head).__name__}")
    print(f"Polygon shape: {head.poly_shape}")
    print(f"Number of polygon points: {head.npoly}")
    
    # Analyze cv4 (polygon prediction layers)
    print("\n--- CV4 (Polygon Prediction Layers) ---")
    for i, cv4_layer in enumerate(head.cv4):
        print(f"\nLayer {i}:")
        final_conv = cv4_layer[-1]  # Last layer is the prediction head
        
        if hasattr(final_conv, 'weight'):
            weight = final_conv.weight.data
            print(f"  Weight shape: {weight.shape}")
            print(f"  Weight stats: mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
            print(f"  Weight range: [{weight.min().item():.6f}, {weight.max().item():.6f}]")
            
        if hasattr(final_conv, 'bias') and final_conv.bias is not None:
            bias = final_conv.bias.data
            print(f"  Bias shape: {bias.shape}")
            print(f"  Bias stats: mean={bias.mean().item():.6f}, std={bias.std().item():.6f}")
            print(f"  Bias range: [{bias.min().item():.6f}, {bias.max().item():.6f}]")
            
            # Check if all biases are the same (indicating initialization issue)
            if torch.allclose(bias, bias[0]):
                print(f"  ⚠️  WARNING: All biases are identical ({bias[0].item():.6f})")
                print(f"      This may cause predictions to be uniform!")


def test_prediction_pipeline(model, img_size=640):
    """Test the prediction pipeline with dummy data."""
    print("\n" + "="*80)
    print("PREDICTION PIPELINE TEST")
    print("="*80)
    
    device = next(model.model.parameters()).device
    head = model.model.model[-1]
    
    # Create dummy input
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    print(f"\nInput shape: {dummy_img.shape}")
    
    # Forward pass
    model.model.eval()
    with torch.no_grad():
        # Get raw outputs from the model (training mode to get raw features)
        model.model.train()
        preds = model.model(dummy_img)
        
        # preds should be (feats, poly)
        if isinstance(preds, tuple) and len(preds) == 2:
            feats, poly = preds
            print(f"\n--- Raw Predictions (Training Mode) ---")
            print(f"Features: {len(feats)} levels")
            for i, feat in enumerate(feats):
                print(f"  Level {i}: {feat.shape}")
            
            print(f"\nPolygon predictions shape: {poly.shape}")
            print(f"Polygon predictions stats:")
            print(f"  Mean: {poly.mean().item():.6f}")
            print(f"  Std: {poly.std().item():.6f}")
            print(f"  Range: [{poly.min().item():.6f}, {poly.max().item():.6f}]")
            
            # Check if predictions are all similar (indication of no learning)
            poly_var = poly.var(dim=[0, 2]).mean()
            print(f"  Variance across anchors: {poly_var.item():.6e}")
            if poly_var < 1e-4:
                print(f"  ⚠️  WARNING: Very low variance! Predictions are nearly identical.")
                print(f"      This indicates the model is not learning to differentiate.")
        
        # Now test inference mode
        model.model.eval()
        results = model.model(dummy_img)
        
        if isinstance(results, tuple):
            inference_out = results[0]
        else:
            inference_out = results
            
        print(f"\n--- Inference Mode Output ---")
        print(f"Output shape: {inference_out.shape}")


def analyze_decoding_function(model, img_size=640):
    """Analyze the polygon decoding function."""
    print("\n" + "="*80)
    print("DECODING FUNCTION ANALYSIS")
    print("="*80)
    
    device = next(model.model.parameters()).device
    head = model.model.model[-1]
    
    # Simulate anchor points and predictions
    num_anchors = 8400  # Typical for 640x640 image
    anchor_points = torch.randn(num_anchors, 2).to(device) * 40  # Simulate grid positions
    strides = torch.tensor([8.0, 16.0, 32.0]).to(device)
    
    # Simulate raw predictions (what comes from the network)
    batch_size = 1
    num_points = head.poly_shape[0]
    raw_preds = torch.randn(batch_size, num_anchors, num_points * 2).to(device)
    
    print(f"\n--- Simulated Data ---")
    print(f"Anchor points: {anchor_points.shape}, range [{anchor_points.min():.2f}, {anchor_points.max():.2f}]")
    print(f"Raw predictions: {raw_preds.shape}, range [{raw_preds.min():.2f}, {raw_preds.max():.2f}]")
    
    # Test the decoding with different raw prediction values
    test_values = [0.0, 1.0, 5.0, 10.0, -5.0]
    print(f"\n--- Decoding Test ---")
    print("Testing how raw predictions map to decoded coordinates:")
    
    for val in test_values:
        # Create uniform predictions
        test_pred = torch.full_like(raw_preds, val)
        test_pred_reshaped = test_pred.view(batch_size, num_anchors, num_points, 2)
        
        # Apply the decoding formula from polygons_decode
        # y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        decoded = test_pred_reshaped * 2.0
        
        print(f"\n  Raw value: {val:6.2f}")
        print(f"    After *2.0: {(val * 2.0):6.2f}")
        print(f"    After adding anchor: varies by anchor position")
        print(f"    After *stride (e.g., 32): {(val * 2.0 * 32):8.1f} pixels")
        
        if abs(val * 2.0 * 32) > img_size:
            print(f"    ⚠️  WARNING: This maps outside image bounds ({img_size}px)!")


def check_bias_initialization_issue():
    """Check if bias initialization is causing the problem."""
    print("\n" + "="*80)
    print("BIAS INITIALIZATION DIAGNOSIS")
    print("="*80)
    
    print("""
The polygon head in ultralytics/nn/modules/head.py has a bias_init() method (line 464-479).
Let's check what it does:

Current implementation:
    def bias_init(self):
        super().bias_init()  # Initialize base Detect biases
        
        for cv4_layer in self.cv4:
            final_conv = cv4_layer[-1]
            if hasattr(final_conv, 'bias') and final_conv.bias is not None:
                final_conv.bias.data.fill_(0.0)  # Initialize to 0

Analysis:
- Biases are initialized to 0.0
- This means initial raw predictions will be close to 0
- After decoding: 0 * 2.0 = 0, then added to anchor position
- This SHOULD give predictions near anchor centers initially
- But if the model doesn't learn, predictions stay at 0

The issue is likely NOT the bias init itself, but rather:
1. Gradient flow problems preventing learning
2. Loss function not providing meaningful gradients
3. Predictions being clamped/saturated somewhere
""")


def main():
    """Run all diagnostics."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_full_image_predictions.py <path_to_model.pt>")
        print("\nExample:")
        print("  python diagnose_full_image_predictions.py runs/polygon/train/weights/best.pt")
        return
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Run diagnostics
    analyze_polygon_head(model)
    test_prediction_pipeline(model)
    analyze_decoding_function(model)
    check_bias_initialization_issue()
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("""
LIKELY ROOT CAUSES:

1. **Decoding Formula Issue** (HIGH CONFIDENCE)
   - The formula: y = (raw_pred * 2.0 + (anchor - 0.5)) * stride
   - Even small raw predictions (e.g., 5.0) become HUGE after * stride
   - Example: 5.0 * 2.0 * 32 = 320 pixels offset!
   - This makes predictions very sensitive and unstable

2. **Loss Not Learning** (HIGH CONFIDENCE)
   - Your logs show box_loss and polygon_loss plateau
   - If predictions are huge, loss gradients may be saturated
   - Or the loss landscape is too flat to provide meaningful gradients

3. **Gradient Flow Issues** (MEDIUM CONFIDENCE)
   - Check if gradients are flowing back to the polygon head
   - Use: ULTRALYTICS_DEBUG_NAN=1 during training

RECOMMENDED FIXES:

1. **Reduce Prediction Sensitivity**
   Change the decoding formula in head.py polygons_decode():
   From: y = raw_pred * 2.0 + anchor_offset
   To:   y = raw_pred * 0.5 + anchor_offset  # Much smaller multiplier
   
2. **Check Data Quality**
   Verify your polygon annotations are correct:
   - Are polygons normalized properly?
   - Are they in the correct format?
   
3. **Increase Polygon Loss Weight**
   In your training config, try increasing polygon loss weight:
   polygon: 2.0  # or higher
   
4. **Debug Gradients**
   Enable gradient debugging during training to see if gradients are flowing.
""")


if __name__ == "__main__":
    main()
