#!/usr/bin/env python3
"""
Safe training script for polygon detection with NaN prevention.
"""

from ultralytics import YOLO
import torch

# Optional: Enable NaN debugging
import os
# os.environ["ULTRALYTICS_DEBUG_NAN"] = "1"  # Uncomment to debug NaN issues

def train_safe():
    """Train with safety mechanisms to prevent NaN."""
    
    # Load model
    model = YOLO('yolo11n-polygon.yaml')
    
    # Train with safety settings
    results = model.train(
        data='your_polygon_data.yaml',  # UPDATE THIS PATH
        epochs=100,
        imgsz=640,
        batch=16,
        
        # === NaN PREVENTION SETTINGS ===
        gradient_clip=1.0,     # Clip gradients to prevent explosion
        lr0=0.005,             # Conservative initial learning rate  
        lrf=0.1,               # Final learning rate = lr0 * lrf
        amp=False,             # Disable mixed precision for stability
        
        # === OPTIONAL STABILITY SETTINGS ===
        patience=50,           # Early stopping if no improvement
        warmup_epochs=3,       # Gradual learning rate warmup
        # optimizer='AdamW',   # Try different optimizer if needed
        
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
        print("\nTraining completed successfully!")
    except RuntimeError as e:
        if "NaN" in str(e):
            print("\n" + "=" * 60)
            print("NaN ERROR DETECTED!")
            print("=" * 60)
            print(f"Error: {e}")
            print()
            print("NEXT STEPS:")
            print("1. Enable debug mode: Uncomment os.environ line at top")
            print("2. Lower learning rate even more (lr0=0.001)")
            print("3. Check if you have corrupt checkpoint weights")
            print("4. Try starting from scratch without resume=True")
        raise
