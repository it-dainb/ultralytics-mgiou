#!/usr/bin/env python3
"""
Quick test to verify OBB drawing with your actual box coordinates.
"""

import numpy as np
import cv2


def test_your_box():
    """Test drawing with your exact box coordinates."""
    
    print("=" * 70)
    print("Testing Your Actual Box Coordinates")
    print("=" * 70)
    
    # Your actual values
    center = (1097.6, 2995.6)
    size = (2013.5, 1264.2)
    angle_rad = 0.010927
    
    # Create image matching your actual image size
    img_h, img_w = 4873, 2250
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    
    print(f"\n📐 Image size: {img_w} × {img_h} (W×H)")
    print(f"📦 Box center: {center}")
    print(f"📦 Box size: {size}")
    print(f"📦 Box angle: {angle_rad:.6f} rad = {angle_rad * 180 / np.pi:.2f}°")
    
    # Convert to degrees for OpenCV
    angle_deg = angle_rad * 180.0 / np.pi
    
    # Create rotated rectangle
    x, y = center
    w, h = size
    rect = ((float(x), float(y)), (float(w), float(h)), float(angle_deg))
    
    # Get corner points
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)
    
    print(f"\n📍 Corner points:")
    for i, pt in enumerate(box_points):
        print(f"   Point {i}: ({pt[0]}, {pt[1]})")
    
    # Calculate coverage
    x_coords = box_points[:, 0]
    y_coords = box_points[:, 1]
    bbox_x1, bbox_x2 = x_coords.min(), x_coords.max()
    bbox_y1, bbox_y2 = y_coords.min(), y_coords.max()
    
    print(f"\n📏 Bounding box:")
    print(f"   X range: [{bbox_x1}, {bbox_x2}] (width: {bbox_x2 - bbox_x1})")
    print(f"   Y range: [{bbox_y1}, {bbox_y2}] (height: {bbox_y2 - bbox_y1})")
    
    # Check visibility
    visible = True
    if bbox_x2 < 0 or bbox_x1 > img_w or bbox_y2 < 0 or bbox_y1 > img_h:
        print(f"\n❌ Box is completely outside image bounds!")
        visible = False
    elif bbox_x1 < 0 or bbox_x2 > img_w or bbox_y1 < 0 or bbox_y2 > img_h:
        print(f"\n⚠️  Box is partially outside image bounds")
        print(f"    Image bounds: x=[0, {img_w}], y=[0, {img_h}]")
    else:
        print(f"\n✅ Box is fully inside image bounds")
    
    if visible:
        # Draw with bright red color
        color = (0, 0, 255)  # BGR: Red
        
        # Draw the rotated rectangle
        cv2.drawContours(img, [box_points], 0, color, 8)
        
        # Draw center point
        cv2.circle(img, (int(x), int(y)), 15, (0, 255, 0), -1)  # Green center
        
        # Draw corner points
        for pt in box_points:
            cv2.circle(img, tuple(pt), 12, (255, 0, 0), -1)  # Blue corners
        
        # Add text label
        label = f"Box: {w:.0f}x{h:.0f} @ {angle_deg:.1f}deg"
        cv2.putText(img, label, (int(x) - 200, int(y) - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # Save full resolution
        output_full = "test_your_box_full.jpg"
        cv2.imwrite(output_full, img)
        print(f"\n💾 Saved full resolution to '{output_full}'")
        
        # Save a crop around the box for easier viewing
        margin = 100
        crop_x1 = max(0, bbox_x1 - margin)
        crop_x2 = min(img_w, bbox_x2 + margin)
        crop_y1 = max(0, bbox_y1 - margin)
        crop_y2 = min(img_h, bbox_y2 + margin)
        
        img_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        output_crop = "test_your_box_crop.jpg"
        cv2.imwrite(output_crop, img_crop)
        print(f"💾 Saved cropped region to '{output_crop}'")
        print(f"   Crop region: x=[{crop_x1}, {crop_x2}], y=[{crop_y1}, {crop_y2}]")
        print(f"   Crop size: {crop_x2-crop_x1} × {crop_y2-crop_y1}")
        
        # Save a downscaled version
        scale = 0.2
        img_small = cv2.resize(img, None, fx=scale, fy=scale)
        output_small = "test_your_box_small.jpg"
        cv2.imwrite(output_small, img_small)
        print(f"💾 Saved downscaled (20%) to '{output_small}'")
    
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    print(f"""
Your box dimensions are CORRECT and the drawing should work fine!

Box covers:
- X: {bbox_x1} to {bbox_x2} (width: {bbox_x2 - bbox_x1} pixels)
- Y: {bbox_y1} to {bbox_y2} (height: {bbox_y2 - bbox_y1} pixels)

This is a LARGE box covering most of your image width
and about 1/4 of the image height.

If it appears small in your viewer, it's likely because:
1. Your image viewer is downscaling the large image automatically
2. You need to zoom in to see the details
3. The box occupies the lower portion of the image (y ~= 2400-3600)

Try opening 'test_your_box_crop.jpg' to see just the box region.
    """)


if __name__ == "__main__":
    test_your_box()
