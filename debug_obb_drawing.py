#!/usr/bin/env python3
"""
Debug script to test OBB drawing with known box values.
This helps identify if the issue is with drawing or with inference.
"""

import numpy as np
import cv2


def test_rotated_rectangle_drawing():
    """Test drawing rotated rectangles with known values."""
    
    print("=" * 70)
    print("Testing Rotated Rectangle Drawing")
    print("=" * 70)
    
    # Create a blank image
    img_size = (4000, 6000, 3)  # H, W, C - typical aerial image size
    img = np.ones(img_size, dtype=np.uint8) * 255
    
    print(f"\n📐 Image size: {img.shape[1]}×{img.shape[0]} (W×H)")
    
    # Test case 1: Your actual box
    test_boxes = [
        {
            "name": "Your actual box",
            "center": (1097.6, 2995.6),
            "size": (2013.5, 1264.2),
            "angle_rad": 0.010927,
            "color": (0, 0, 255)  # Red
        },
        {
            "name": "Horizontal box (0°)",
            "center": (3000, 2000),
            "size": (1000, 500),
            "angle_rad": 0.0,
            "color": (0, 255, 0)  # Green
        },
        {
            "name": "45° rotated box",
            "center": (3000, 1000),
            "size": (1000, 500),
            "angle_rad": np.pi / 4,
            "color": (255, 0, 0)  # Blue
        },
        {
            "name": "90° rotated box",
            "center": (5000, 2000),
            "size": (1000, 500),
            "angle_rad": np.pi / 2,
            "color": (255, 255, 0)  # Cyan
        }
    ]
    
    for test in test_boxes:
        x, y = test["center"]
        w, h = test["size"]
        angle_rad = test["angle_rad"]
        color = test["color"]
        name = test["name"]
        
        # Convert angle to degrees
        angle_deg = angle_rad * 180.0 / np.pi
        
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"  Center: ({x:.1f}, {y:.1f})")
        print(f"  Size: {w:.1f} × {h:.1f} pixels")
        print(f"  Angle: {angle_rad:.6f} rad = {angle_deg:.2f}°")
        
        # Create rotated rectangle
        rect = ((float(x), float(y)), (float(w), float(h)), float(angle_deg))
        
        # Get corner points
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        
        print(f"  Corner points:")
        for i, pt in enumerate(box_points):
            print(f"    Point {i}: ({pt[0]}, {pt[1]})")
        
        # Calculate bounding box
        x_coords = box_points[:, 0]
        y_coords = box_points[:, 1]
        bbox_x1, bbox_x2 = x_coords.min(), x_coords.max()
        bbox_y1, bbox_y2 = y_coords.min(), y_coords.max()
        bbox_w = bbox_x2 - bbox_x1
        bbox_h = bbox_y2 - bbox_y1
        
        print(f"  Bounding box: ({bbox_x1}, {bbox_y1}) to ({bbox_x2}, {bbox_y2})")
        print(f"  Bounding box size: {bbox_w} × {bbox_h}")
        
        # Check if box is visible in image
        if bbox_x2 < 0 or bbox_y2 < 0 or bbox_x1 > img.shape[1] or bbox_y1 > img.shape[0]:
            print(f"  ⚠️  WARNING: Box is completely outside image!")
        elif bbox_x1 < 0 or bbox_y1 < 0 or bbox_x2 > img.shape[1] or bbox_y2 > img.shape[0]:
            print(f"  ⚠️  WARNING: Box is partially outside image!")
        else:
            print(f"  ✅ Box is fully inside image")
        
        # Draw rotated rectangle
        cv2.drawContours(img, [box_points], 0, color, 5)
        
        # Draw center point
        cv2.circle(img, (int(x), int(y)), 10, color, -1)
        
        # Add label
        label = f"{name}: {w:.0f}x{h:.0f} @ {angle_deg:.1f}°"
        cv2.putText(img, label, (int(x) - 200, int(y) - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Save result
    output_path = "debug_obb_drawing.jpg"
    cv2.imwrite(output_path, img)
    print(f"\n💾 Saved test image to '{output_path}'")
    
    # Also save a scaled-down version for easier viewing
    scale = 0.25
    img_small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    output_path_small = "debug_obb_drawing_small.jpg"
    cv2.imwrite(output_path_small, img_small)
    print(f"💾 Saved scaled version to '{output_path_small}' (25% size)")
    
    print("\n" + "=" * 70)
    print("Test complete! Check the output images.")
    print("=" * 70)


def test_opencv_angle_interpretation():
    """Test how OpenCV interprets angles in RotatedRect."""
    
    print("\n" + "=" * 70)
    print("Testing OpenCV Angle Interpretation")
    print("=" * 70)
    
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # Test different angle values
    test_angles_deg = [0, 30, 45, 60, 90, 120, 135, 150, 180, -45, -90]
    
    for i, angle_deg in enumerate(test_angles_deg):
        x = 200 + (i % 4) * 250
        y = 200 + (i // 4) * 250
        
        rect = ((x, y), (150, 80), angle_deg)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        
        # Draw
        cv2.drawContours(img, [box_points], 0, (0, 0, 255), 2)
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        cv2.putText(img, f"{angle_deg}°", (x - 30, y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    output_path = "debug_angle_interpretation.jpg"
    cv2.imwrite(output_path, img)
    print(f"\n💾 Saved angle test to '{output_path}'")
    print("\nNote: OpenCV RotatedRect uses clockwise rotation from horizontal axis")
    print("      angle = 0°: width is horizontal")
    print("      angle > 0: rotates clockwise")


if __name__ == "__main__":
    test_rotated_rectangle_drawing()
    test_opencv_angle_interpretation()
    
    print("\n" + "=" * 70)
    print("Diagnostic Tips:")
    print("=" * 70)
    print("""
1. Check if box coordinates are in correct range for your image
2. Verify angle is in correct format (radians vs degrees)
3. Ensure width/height are positive and reasonable
4. Check if box center is inside or near image bounds
5. Look at the actual corner points to see where box is drawn

If your box dimensions are correct (2013.5 × 1264.2) but appears small:
- The box might be outside the visible image area
- The angle might be causing incorrect rotation
- There might be a scaling issue in preprocessing
    """)
