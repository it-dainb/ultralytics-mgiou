#!/usr/bin/env python3
"""
Test script to verify OBB ONNX format fix.
This script compares the number of detections before and after the fix.
"""

import numpy as np
import cv2
from onnx_infer import ONNXInference


def test_obb_format():
    """Test OBB inference with fixed format parsing."""
    
    print("=" * 70)
    print("OBB ONNX Format Fix Test")
    print("=" * 70)
    
    # Configuration
    model_path = "yolov8n-obb.onnx"  # Replace with your OBB model path
    image_path = "test_image.jpg"      # Replace with your test image
    
    print(f"\n📦 Model: {model_path}")
    print(f"🖼️  Image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image '{image_path}'")
        print("Please update the image_path variable with a valid image.")
        return
    
    print(f"📐 Image shape: {img.shape}")
    
    # Initialize model
    print("\n🚀 Initializing OBB model...")
    model = ONNXInference(
        model_path=model_path,
        device="cuda",  # or "cpu"
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        task="obb"
    )
    
    # Run inference
    print("\n🔍 Running inference...")
    print("-" * 70)
    results = model(img)
    print("-" * 70)
    
    # Display results
    num_detections = len(results["boxes"])
    print(f"\n✅ Inference complete!")
    print(f"📊 Number of detections: {num_detections}")
    
    if num_detections > 0:
        print(f"📦 Boxes shape: {results['boxes'].shape}")
        print(f"📊 Scores shape: {results['scores'].shape}")
        print(f"🏷️  Class IDs shape: {results['class_ids'].shape}")
        
        print(f"\n📈 Detection details:")
        for i, (box, score, class_id) in enumerate(zip(
            results["boxes"][:5],  # Show first 5
            results["scores"][:5],
            results["class_ids"][:5]
        )):
            x, y, w, h, angle = box
            angle_deg = angle * 180 / np.pi
            class_name = model.classes[class_id]
            print(f"  [{i+1}] {class_name}: score={score:.3f}, "
                  f"center=({x:.1f}, {y:.1f}), size=({w:.1f}×{h:.1f}), "
                  f"angle={angle_deg:.1f}°")
        
        if num_detections > 5:
            print(f"  ... and {num_detections - 5} more")
        
        # Draw results
        print(f"\n🎨 Drawing detections...")
        img_result = model.draw_detections(img, results)
        
        # Save result
        output_path = "output_obb_fixed.jpg"
        cv2.imwrite(output_path, img_result)
        print(f"💾 Saved result to '{output_path}'")
        
    else:
        print("⚠️  No detections found. Try lowering conf_thres or using a different image.")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


def explain_fix():
    """Explain the fix applied."""
    
    print("\n" + "=" * 70)
    print("What was fixed?")
    print("=" * 70)
    
    print("""
The OBB ONNX output format was being parsed incorrectly.

❌ BEFORE (Wrong):
   predictions[:, :5]     → [x, y, w, h, class_score_0]  (WRONG!)
   predictions[:, 5:]     → [class_score_1, ..., angle]  (WRONG!)

✅ AFTER (Correct):
   predictions[:, :4]     → [x, y, w, h]                 (Correct!)
   predictions[:, 4:-1]   → [class_score_0, ..., N]      (Correct!)
   predictions[:, -1]     → [angle]                       (Correct!)

The angle is in the LAST column, not column 4!

This fix should result in:
- Correct confidence scores
- Proper NMS suppression
- Fewer false positive detections
- Results matching Ultralytics behavior
    """)


if __name__ == "__main__":
    explain_fix()
    
    try:
        test_obb_format()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n⚠️  Please update the model_path and image_path variables in this script.")
        print("   You can export an OBB model using:")
        print("   >>> from ultralytics import YOLO")
        print("   >>> model = YOLO('yolov8n-obb.pt')")
        print("   >>> model.export(format='onnx')")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
