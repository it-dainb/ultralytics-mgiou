#!/usr/bin/env python3
"""
Example usage of onnx_infer.py for OBB (Oriented Bounding Box) detection
"""

import cv2
import numpy as np
from onnx_infer import ONNXInference

def example_obb_inference():
    """Example: Run OBB inference on an image."""
    
    # Initialize OBB model
    print("🚀 Initializing OBB model...")
    model = ONNXInference(
        model_path="yolov8n-obb.onnx",  # Replace with your OBB model path
        device="cuda",  # or "cpu"
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        task="obb"  # Important: set task to "obb"
    )
    
    # Load image
    print("📸 Loading image...")
    img = cv2.imread("aerial_image.jpg")  # Replace with your image
    
    if img is None:
        print("❌ Could not load image!")
        return
    
    print(f"📐 Image shape: {img.shape}")
    
    # Run inference
    print("🔍 Running inference...")
    results = model(img)
    
    # Print results
    num_detections = len(results["boxes"])
    print(f"\n🎯 Found {num_detections} detections")
    
    if num_detections > 0:
        print("\nDetailed results:")
        for i, (box, score, class_id) in enumerate(
            zip(results["boxes"], results["scores"], results["class_ids"])
        ):
            x, y, w, h, angle = box
            angle_deg = angle * 180.0 / np.pi
            print(f"  {i + 1}. {model.classes[class_id]}:")
            print(f"      Confidence: {score:.3f}")
            print(f"      Center: ({x:.1f}, {y:.1f})")
            print(f"      Size: {w:.1f} x {h:.1f}")
            print(f"      Angle: {angle_deg:.1f}°")
    
    # Draw detections
    print("\n🎨 Drawing detections...")
    img_result = model.draw_detections(img, results)
    
    # Save result
    output_path = "output_obb.jpg"
    cv2.imwrite(output_path, img_result)
    print(f"💾 Saved result to: {output_path}")
    
    # Optional: Display
    # cv2.imshow("OBB Detection", img_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def example_regular_detection():
    """Example: Run regular detection on an image."""
    
    # Initialize regular detection model
    print("🚀 Initializing detection model...")
    model = ONNXInference(
        model_path="yolov8n.onnx",  # Replace with your model path
        device="cuda",  # or "cpu"
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        task="detect"  # Regular detection
    )
    
    # Load image
    print("📸 Loading image...")
    img = cv2.imread("image.jpg")  # Replace with your image
    
    if img is None:
        print("❌ Could not load image!")
        return
    
    # Run inference
    print("🔍 Running inference...")
    results = model(img)
    
    # Print results
    num_detections = len(results["boxes"])
    print(f"\n🎯 Found {num_detections} detections")
    
    if num_detections > 0:
        for i, (box, score, class_id) in enumerate(
            zip(results["boxes"], results["scores"], results["class_ids"])
        ):
            x1, y1, x2, y2 = box
            print(f"  {i + 1}. {model.classes[class_id]}: {score:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
    # Draw and save
    img_result = model.draw_detections(img, results)
    cv2.imwrite("output_detect.jpg", img_result)
    print("💾 Saved result to: output_detect.jpg")


def compare_tasks():
    """Compare outputs between regular detection and OBB."""
    
    print("=" * 60)
    print("COMPARISON: Regular Detection vs OBB")
    print("=" * 60)
    
    # This shows the difference in output format
    
    print("\n1️⃣ REGULAR DETECTION")
    print("   Box format: [x1, y1, x2, y2] (axis-aligned)")
    print("   Use for: standard objects, street scenes, indoor scenes")
    
    print("\n2️⃣ OBB (Oriented Bounding Box)")
    print("   Box format: [x_center, y_center, width, height, angle]")
    print("   Use for: aerial imagery, rotated objects, satellite data")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("ONNX Inference Examples")
    print("=" * 60)
    
    # Show comparison
    compare_tasks()
    
    print("\n\nChoose example to run:")
    print("1. Regular Detection")
    print("2. OBB Detection")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            example_regular_detection()
        elif choice == "2":
            example_obb_inference()
        elif choice == "3":
            print("\n--- REGULAR DETECTION ---")
            example_regular_detection()
            print("\n--- OBB DETECTION ---")
            example_obb_inference()
        else:
            print("Invalid choice. Run script with --help for usage.")
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
