# ONNX Inference - Standalone Script

Standalone ONNX inference script for YOLO models, extracted from Ultralytics AutoBackend.

## Features

✅ **No Ultralytics Required** - Pure NumPy + OpenCV + ONNX Runtime  
✅ **Object Detection** - Regular bounding boxes  
✅ **OBB (Oriented Bounding Boxes)** - Rotated bounding boxes for aerial/satellite imagery  
✅ **CPU & CUDA Support** - Automatic device detection  
✅ **NMS Included** - Non-Maximum Suppression for both regular and rotated boxes  

## Installation

```bash
pip install numpy opencv-python onnxruntime
# OR for GPU support
pip install numpy opencv-python onnxruntime-gpu
```

## Usage

### Object Detection (Regular Bounding Boxes)

```bash
python onnx_infer.py \
    --model yolov8n.onnx \
    --source image.jpg \
    --task detect \
    --conf 0.25 \
    --iou 0.45 \
    --device cuda \
    --show
```

### OBB (Oriented Bounding Boxes)

For aerial/satellite imagery with rotated objects:

```bash
python onnx_infer.py \
    --model yolov8n-obb.onnx \
    --source aerial_image.jpg \
    --task obb \
    --conf 0.25 \
    --iou 0.45 \
    --device cuda \
    --show
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | *required* | Path to ONNX model file |
| `--source` | str | *required* | Path to input image |
| `--output` | str | `output.jpg` | Path to save output image |
| `--task` | str | `detect` | Task type: `detect` or `obb` |
| `--device` | str | `cpu` | Device: `cpu` or `cuda` |
| `--img-size` | int | `640` | Input image size |
| `--conf` | float | `0.25` | Confidence threshold |
| `--iou` | float | `0.45` | IoU threshold for NMS |
| `--max-det` | int | `300` | Maximum detections |
| `--show` | flag | `False` | Display results window |

## Python API

```python
import cv2
from onnx_infer import ONNXInference

# Initialize model for regular detection
model = ONNXInference(
    model_path="yolov8n.onnx",
    device="cuda",
    conf_thres=0.25,
    iou_thres=0.45,
    task="detect"  # or "obb" for oriented bounding boxes
)

# Load image
img = cv2.imread("image.jpg")

# Run inference
results = model(img)

# Results format:
# results = {
#     "boxes": np.ndarray,      # For detect: [x1, y1, x2, y2]
#                               # For OBB: [x_center, y_center, w, h, angle_rad]
#     "scores": np.ndarray,     # Confidence scores
#     "class_ids": np.ndarray   # Class IDs
# }

# Draw results
img_result = model.draw_detections(img, results)

# Save
cv2.imwrite("output.jpg", img_result)
```

## OBB Format

### ONNX Output Format
The raw ONNX model output for OBB has the shape `[batch, features, num_predictions]` which gets transposed to `[batch, num_predictions, features]`.

Each prediction has the format: `[x_center, y_center, width, height, class_score_0, class_score_1, ..., class_score_N, angle]`

- **x_center, y_center**: Center coordinates of the rotated box (before padding removal/scaling)
- **width, height**: Box dimensions (before scaling)
- **class_score_0...N**: Confidence scores for each class
- **angle**: Rotation angle in **radians**, positioned as the **LAST column** (after all class scores)
  - Typical range: [-π/4, 3π/4] (via `(sigmoid - 0.25) * π` transformation in model)

### After Postprocessing
After NMS and coordinate transformation, boxes are in the format: `[x_center, y_center, width, height, angle]`

- **x_center, y_center**: Center coordinates in original image space
- **width, height**: Box dimensions scaled to original image size
- **angle**: Rotation angle in **radians** (unchanged from model output)

The visualization automatically converts to OpenCV's rotated rectangle format.

## Default Classes

### Object Detection (COCO)
80 classes including: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, etc.

### OBB (DOTA)
15 classes for aerial imagery:
- plane, ship, storage tank, baseball diamond, tennis court
- basketball court, ground track field, harbor, bridge
- large vehicle, small vehicle, helicopter, roundabout
- soccer ball field, swimming pool

## Export ONNX Models

To export YOLO models to ONNX format:

```python
from ultralytics import YOLO

# Regular detection
model = YOLO("yolov8n.pt")
model.export(format="onnx")

# OBB
model = YOLO("yolov8n-obb.pt")
model.export(format="onnx")
```

## Performance Tips

1. **Use CUDA** for faster inference on GPU
2. **Adjust img-size** - larger = better accuracy, slower speed
3. **Tune confidence** threshold based on your use case
4. **Batch processing** - modify script to process multiple images

## Limitations

- OBB NMS uses simplified IoU calculation for speed
- For production OBB applications, consider using `cv2.rotatedRectangleIntersection` for more accurate IoU
- Single image processing (can be extended for video/batch)

## License

AGPL-3.0 License - Same as Ultralytics YOLO
