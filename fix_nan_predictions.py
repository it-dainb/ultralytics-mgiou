#!/usr/bin/env python3
"""
Script to diagnose and fix NaN in polygon predictions during training.

The error "RuntimeError: NaN detected in pred_kpts at PolygonLoss.forward input"
indicates that the model's predictions contain NaN BEFORE reaching the loss function.

This script helps diagnose the root cause and provides solutions.
"""

import torch
import os
import sys

def diagnose_nan_source():
    """Diagnose where NaN is coming from in the training pipeline."""
    
    print("=" * 80)
    print("NaN IN POLYGON PREDICTIONS - DIAGNOSTIC GUIDE")
    print("=" * 80)
    print()
    
    print("ERROR LOCATION:")
    print("  File: ultralytics/utils/loss.py")
    print("  Line: 839")
    print("  Message: RuntimeError: NaN detected in pred_kpts at PolygonLoss.forward input")
    print("  Shape: torch.Size([10, 4, 2])")
    print("  NaN count: 80")
    print()
    
    print("ROOT CAUSE:")
    print("  The NaN is in the model's RAW PREDICTIONS (pred_poly), not in the loss function.")
    print("  This means the neural network is outputting NaN values.")
    print()
    
    print("COMMON CAUSES:")
    print("  1. Gradient Explosion - Gradients became too large and corrupted weights")
    print("  2. Bad Initialization - Model weights initialized with invalid values")
    print("  3. Learning Rate Too High - Optimizer took steps that destabilized training")
    print("  4. Loss Spike - A previous batch had extreme loss causing gradient explosion")
    print("  5. Mixed Precision Issues - FP16 training caused overflow")
    print()
    
    print("=" * 80)
    print("SOLUTION 1: Enable Gradient Clipping (RECOMMENDED)")
    print("=" * 80)
    print()
    print("Add gradient clipping to prevent gradient explosion:")
    print()
    print("In your training script or config:")
    print("  model.train(")
    print("      data='your_data.yaml',")
    print("      epochs=100,")
    print("      gradient_clip=1.0,  # <-- Add this line")
    print("  )")
    print()
    print("Or in your training config YAML:")
    print("  gradient_clip: 1.0")
    print()
    
    print("=" * 80)
    print("SOLUTION 2: Reduce Learning Rate")
    print("=" * 80)
    print()
    print("Lower the learning rate to stabilize training:")
    print()
    print("  model.train(")
    print("      data='your_data.yaml',")
    print("      lr0=0.001,  # Reduce from default 0.01")
    print("  )")
    print()
    
    print("=" * 80)
    print("SOLUTION 3: Check for Corrupt Weights")
    print("=" * 80)
    print()
    print("If you resumed training from a checkpoint, the weights may be corrupted.")
    print("Try restarting training from a fresh model:")
    print()
    print("  # Don't resume - start fresh")
    print("  model = YOLO('yolo11n-polygon.yaml')")
    print("  model.train(data='your_data.yaml', epochs=100)")
    print()
    
    print("=" * 80)
    print("SOLUTION 4: Enable NaN Detection to Find Exact Source")
    print("=" * 80)
    print()
    print("Run with debug mode to see WHERE the NaN first appears:")
    print()
    print("  export ULTRALYTICS_DEBUG_NAN=1")
    print("  python your_training_script.py")
    print()
    print("This will show whether NaN appears in:")
    print("  - pred_poly (raw) - from the network output")
    print("  - pred_poly (decoded) - from the decode operation")
    print()
    
    print("=" * 80)
    print("SOLUTION 5: Check Mixed Precision Settings")
    print("=" * 80)
    print()
    print("If using FP16/mixed precision, try disabling it:")
    print()
    print("  model.train(")
    print("      data='your_data.yaml',")
    print("      amp=False,  # Disable automatic mixed precision")
    print("  )")
    print()
    
    print("=" * 80)
    print("IMMEDIATE ACTION (Recovery)")
    print("=" * 80)
    print()
    print("If training has already crashed:")
    print()
    print("1. Find the last good checkpoint BEFORE NaN appeared:")
    print("   ls -lth runs/detect/train/weights/")
    print()
    print("2. Resume from an EARLIER checkpoint (not the latest):")
    print("   model = YOLO('runs/detect/train/weights/epoch50.pt')  # Use earlier epoch")
    print("   model.train(")
    print("       data='your_data.yaml',")
    print("       resume=False,  # Don't resume - restart with new settings")
    print("       gradient_clip=1.0,  # Add safety")
    print("       lr0=0.001,  # Lower learning rate")
    print("   )")
    print()
    
    print("=" * 80)
    print("COMPLETE EXAMPLE - Safe Training Configuration")
    print("=" * 80)
    print()
    print("from ultralytics import YOLO")
    print()
    print("model = YOLO('yolo11n-polygon.yaml')")
    print("model.train(")
    print("    data='polygon_data.yaml',")
    print("    epochs=100,")
    print("    # Stability settings")
    print("    gradient_clip=1.0,      # Prevent gradient explosion")
    print("    lr0=0.005,              # Conservative learning rate")
    print("    lrf=0.1,                # Final learning rate factor")
    print("    amp=False,              # Disable mixed precision if unstable")
    print("    # Optional: Early stopping")
    print("    patience=50,            # Stop if no improvement")
    print(")")
    print()
    
    print("=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print()
    print("After applying fixes, training should:")
    print("  ✓ Not crash with NaN errors")
    print("  ✓ Show steady loss decrease")
    print("  ✓ Have gradient norms < 100 (check with gradient_clip logging)")
    print()
    print("If NaN still appears, run with ULTRALYTICS_DEBUG_NAN=1 to see exact location")
    print()

def create_safe_training_script():
    """Create a safe training script with all protections enabled."""
    
    script_content = """#!/usr/bin/env python3
'''
Safe training script for polygon detection with NaN prevention.
'''

from ultralytics import YOLO
import torch

# Optional: Enable NaN debugging
import os
os.environ["ULTRALYTICS_DEBUG_NAN"] = "0"  # Set to "1" to debug NaN issues

def train_safe():
    '''Train with safety mechanisms to prevent NaN.'''
    
    # Load model
    model = YOLO('yolo11n-polygon.yaml')
    
    # Train with safety settings
    results = model.train(
        data='your_polygon_data.yaml',  # UPDATE THIS
        epochs=100,
        imgsz=640,
        batch=16,
        
        # === NaN PREVENTION SETTINGS ===
        gradient_clip=1.0,     # Clip gradients to prevent explosion
        lr0=0.005,             # Conservative initial learning rate
        lrf=0.1,               # Final learning rate = lr0 * lrf
        amp=False,             # Disable mixed precision for stability
        
        # === OPTIONAL STABILITY SETTINGS ===
        # patience=50,         # Early stopping if no improvement
        # warmup_epochs=3,     # Gradual learning rate warmup
        # optimizer='AdamW',   # Try different optimizer
        
        # === MONITORING ===
        plots=True,            # Generate training plots
        save_period=10,        # Save checkpoint every 10 epochs
    )
    
    return results

if __name__ == '__main__':
    print("Starting safe training with NaN prevention...")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print("=" * 60)
    
    try:
        results = train_safe()
        print("\\nTraining completed successfully!")
    except RuntimeError as e:
        if "NaN" in str(e):
            print("\\n" + "=" * 60)
            print("NaN ERROR DETECTED!")
            print("=" * 60)
            print(f"Error: {e}")
            print()
            print("NEXT STEPS:")
            print("1. Enable debug mode: export ULTRALYTICS_DEBUG_NAN=1")
            print("2. Run: python fix_nan_predictions.py")
            print("3. Check gradient norms in training logs")
            print("4. Try even lower learning rate (lr0=0.001)")
        raise
"""
    
    with open('train_safe_polygon.py', 'w') as f:
        f.write(script_content)
    
    print("Created: train_safe_polygon.py")
    print("Edit the file to set your data path, then run:")
    print("  python train_safe_polygon.py")
    print()

if __name__ == '__main__':
    diagnose_nan_source()
    
    print()
    print("=" * 80)
    print("CREATE SAFE TRAINING SCRIPT?")
    print("=" * 80)
    print()
    response = input("Generate train_safe_polygon.py with all fixes? [y/N]: ").strip().lower()
    
    if response in ['y', 'yes']:
        create_safe_training_script()
    else:
        print()
        print("To create the script manually, run:")
        print("  python fix_nan_predictions.py --create-script")
