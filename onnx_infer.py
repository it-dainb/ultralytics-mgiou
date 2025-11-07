#!/usr/bin/env python3
"""
Standalone ONNX Inference Script for YOLO Models
Extracted from Ultralytics AutoBackend - No Ultralytics installation required

This script provides standalone ONNX inference functionality for YOLO models
without requiring the full Ultralytics package installation.

Supports:
    - Object Detection (detect)
    - Oriented Bounding Boxes (OBB)

Dependencies:
    - numpy
    - opencv-python
    - onnxruntime (or onnxruntime-gpu for CUDA support)

Usage:
    # Regular detection
    python onnx_infer.py --model yolov8n.onnx --source image.jpg --task detect --conf 0.25 --iou 0.45
    
    # OBB detection
    python onnx_infer.py --model yolov8n-obb.onnx --source image.jpg --task obb --conf 0.25 --iou 0.45
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.
    
    This class resizes and pads images to a specified shape while preserving aspect ratio.
    """

    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scaleFill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
    ):
        """
        Initialize LetterBox object for resizing and padding images.

        Args:
            new_shape (tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize.
            scaleFill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the image. If False, place image in top-left corner.
            stride (int): Stride for ensuring image size is divisible by stride.
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
        """
        Resize and pad image while preserving aspect ratio.

        Args:
            image (np.ndarray): Input image as numpy array.

        Returns:
            (np.ndarray): Resized and padded image.
            (tuple[float, float]): Ratio (width, height) used for resizing.
            (tuple[int, int]): Padding (left, top) applied to the image.
        """
        shape = image.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return image, ratio, (left, top)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (np.ndarray): Bounding boxes with shape (N, 4) in format [x1, y1, x2, y2].
        scores (np.ndarray): Confidence scores with shape (N,).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        (list[int]): Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return []

    # Convert to float if needed
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)

    # Sort by score descending
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sorts 4-point polygon in TL, TR, BR, BL order.
    
    Args:
        pts (np.ndarray): Array of 4 points with shape (4, 2).
    
    Returns:
        (np.ndarray): Ordered points [top-left, top-right, bottom-right, bottom-left].
    """
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def rotate_ordered_points(rect: np.ndarray, angle_type: int) -> np.ndarray:
    """
    Rotates TL, TR, BR, BL ordering clockwise by given angle_type.
    
    Args:
        rect (np.ndarray): Ordered points [TL, TR, BR, BL] with shape (4, 2).
        angle_type (int): Rotation angle in {0, 90, 180, 270} degrees.
    
    Returns:
        (np.ndarray): Rotated ordered points.
    """
    k = (angle_type // 90) % 4
    return np.roll(rect, k, axis=0)


def crop_img_from_polygon(
    img: np.ndarray,
    polygon: np.ndarray,
    angle_type: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Crops a perspective-warped region from an image based on 4-point polygon.
    
    Args:
        img (np.ndarray): Input image with shape (H, W, 3).
        polygon (np.ndarray): Polygon coordinates with shape (4, 2).
        angle_type (int): Rotation direction in {0, 90, 180, 270} degrees (clockwise).
    
    Returns:
        (np.ndarray): Warped/cropped image.
        (dict): Dictionary containing transform matrices with keys:
            - 'src': Source points (4, 2)
            - 'dst': Destination points (4, 2)
            - 'M': Perspective transform matrix (3, 3)
    """
    polygon = np.asarray(polygon, dtype=np.float32)
    rect = order_points(polygon)
    rect_rot = rotate_ordered_points(rect, angle_type)
    
    # Calculate width and height of the output image
    width_top = np.linalg.norm(rect_rot[1] - rect_rot[0])
    width_bottom = np.linalg.norm(rect_rot[2] - rect_rot[3])
    height_left = np.linalg.norm(rect_rot[3] - rect_rot[0])
    height_right = np.linalg.norm(rect_rot[2] - rect_rot[1])
    width = max(width_top, width_bottom)
    height = max(height_left, height_right)
    
    # Define destination points for the warped image
    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Get perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect_rot, dst_pts)
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))
    
    return warped, {'src': rect_rot, 'dst': dst_pts, 'M': M}


def batch_probiou(obb1: np.ndarray, obb2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Calculate the prob IoU between oriented bounding boxes using accurate OpenCV intersection.
    
    Args:
        obb1 (np.ndarray): Oriented bounding boxes 1, shape (N, 5) [x, y, w, h, angle].
        obb2 (np.ndarray): Oriented bounding boxes 2, shape (M, 5) [x, y, w, h, angle].
        eps (float): Small value to avoid division by zero.
    
    Returns:
        (np.ndarray): IoU scores, shape (N, M).
    """
    # Get number of boxes
    n = len(obb1)
    m = len(obb2)
    
    # Initialize IoU matrix
    ious = np.zeros((n, m), dtype=np.float32)
    
    # Calculate areas
    area1 = obb1[:, 2] * obb1[:, 3]
    area2 = obb2[:, 2] * obb2[:, 3]
    
    # Convert angle from radians to degrees for OpenCV
    for i in range(n):
        x1, y1, w1, h1, a1 = obb1[i]
        angle1_deg = a1 * 180.0 / np.pi
        rect1 = ((float(x1), float(y1)), (float(w1), float(h1)), float(angle1_deg))
        
        for j in range(m):
            x2, y2, w2, h2, a2 = obb2[j]
            angle2_deg = a2 * 180.0 / np.pi
            rect2 = ((float(x2), float(y2)), (float(w2), float(h2)), float(angle2_deg))
            
            # Calculate intersection using OpenCV
            try:
                intersection_type, intersection_points = cv2.rotatedRectangleIntersection(rect1, rect2)
                
                if intersection_type == cv2.INTERSECT_NONE:
                    inter_area = 0.0
                elif intersection_type == cv2.INTERSECT_FULL:
                    inter_area = min(area1[i], area2[j])
                else:
                    # Calculate polygon area
                    if intersection_points is not None and len(intersection_points) >= 3:
                        inter_area = cv2.contourArea(intersection_points)
                    else:
                        inter_area = 0.0
                
                # Calculate IoU
                union_area = area1[i] + area2[j] - inter_area
                ious[i, j] = inter_area / (union_area + eps)
            except:
                # Fallback to 0 if calculation fails
                ious[i, j] = 0.0
    
    return ious


def nms_rotated(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """
    Perform Non-Maximum Suppression (NMS) on rotated bounding boxes.

    Args:
        boxes (np.ndarray): Rotated bounding boxes with shape (N, 5) in format [x, y, w, h, angle].
        scores (np.ndarray): Confidence scores with shape (N,).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        (list[int]): Indices of boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return []

    # Sort by score descending
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Calculate IoU with remaining boxes using accurate rotated IoU
        ious = batch_probiou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Keep boxes with IoU less than threshold
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class ONNXInference:
    """
    ONNX Inference class for YOLO models.
    
    This class handles ONNX model loading, preprocessing, inference, and postprocessing
    for YOLO object detection and OBB (Oriented Bounding Box) models.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        classes: list[str] | None = None,
        task: str = "detect",
    ):
        """
        Initialize ONNX inference session.

        Args:
            model_path (str): Path to the ONNX model file.
            device (str): Device to run inference on ('cpu' or 'cuda').
            img_size (int): Input image size for the model.
            conf_thres (float): Confidence threshold for detections.
            iou_thres (float): IoU threshold for NMS.
            max_det (int): Maximum number of detections to keep.
            classes (list[str], optional): List of class names.
            task (str): Task type - 'detect' for regular detection, 'obb' for oriented bounding boxes.
        """
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.task = task
        self.classes = classes or self._get_default_classes()

        # Initialize ONNX Runtime session
        self._init_session()

        # Initialize preprocessing
        self.letterbox = LetterBox(new_shape=(self.img_size, self.img_size), auto=False, stride=32)

        print(f"✅ Loaded ONNX model: {model_path}")
        print(f"📊 Input shape: {self.input_shape}")
        print(f"🎯 Output names: {self.output_names}")

    def _init_session(self):
        """Initialize ONNX Runtime inference session."""
        # Set up providers (CUDA or CPU)
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
                print("🚀 Using CUDA for inference")
            else:
                print("⚠️  CUDA requested but not available, using CPU")
                self.device = "cpu"
        else:
            print("🖥️  Using CPU for inference")

        # Create inference session
        self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)

        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [x.name for x in self.session.get_outputs()]

        # Get metadata if available
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.dynamic = isinstance(self.session.get_outputs()[0].shape[0], str)
        self.fp16 = "float16" in self.session.get_inputs()[0].type

        # Update img_size from model if available
        if len(self.input_shape) >= 4:
            self.img_size = self.input_shape[2] if self.input_shape[2] > 0 else self.img_size

    def _get_default_classes(self) -> list[str]:
        """Get default class names based on task."""
        if self.task == "obb":
            # Default DOTA (Dataset for Object Detection in Aerial Images) classes
            return [
                "plane",
                "ship",
                "storage tank",
                "baseball diamond",
                "tennis court",
                "basketball court",
                "ground track field",
                "harbor",
                "bridge",
                "large vehicle",
                "small vehicle",
                "helicopter",
                "roundabout",
                "soccer ball field",
                "swimming pool",
            ]
        else:
            # Default COCO classes
            return [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[float, float], tuple[int, int], tuple[int, int]]:
        """
        Preprocess image for inference.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            (np.ndarray): Preprocessed image tensor.
            (tuple[float, float]): Resize ratio (width, height).
            (tuple[int, int]): Padding (left, top).
            (tuple[int, int]): Original image shape (height, width).
        """
        # Store original shape
        orig_shape = img.shape[:2]

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply letterbox
        img_resized, ratio, pad = self.letterbox(img_rgb)

        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Transpose from HWC to CHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)

        return img_batch, ratio, pad, orig_shape

    def postprocess(
        self,
        outputs: list[np.ndarray],
        ratio: tuple[float, float],
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """
        Postprocess model outputs to get detections.

        Args:
            outputs (list[np.ndarray]): Raw model outputs.
            ratio (tuple[float, float]): Resize ratio (width, height).
            pad (tuple[int, int]): Padding (left, top).
            orig_shape (tuple[int, int]): Original image shape (height, width).

        Returns:
            (dict): Dictionary containing:
                - boxes: Bounding boxes in format [x1, y1, x2, y2] for detect, or [x, y, w, h, angle] for OBB
                - scores: Confidence scores
                - class_ids: Class IDs
        """
        # Get predictions - shape is typically (1, num_predictions, 4/5 + num_classes)
        predictions = outputs[0]

        # Handle different output shapes
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension

        # Transpose if needed (num_predictions, features) -> (features, num_predictions)
        if predictions.shape[1] > predictions.shape[0]:
            predictions = predictions.T

        if self.task == "obb":
            return self._postprocess_obb(predictions, ratio, pad, orig_shape)
        else:
            return self._postprocess_detect(predictions, ratio, pad, orig_shape)

    def _postprocess_detect(
        self,
        predictions: np.ndarray,
        ratio: tuple[float, float],
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Postprocess for regular detection task."""
        # Extract boxes and scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        scores_all = predictions[:, 4:]  # class scores

        # Get max scores and class IDs
        scores = np.max(scores_all, axis=1)
        class_ids = np.argmax(scores_all, axis=1)

        # Filter by confidence threshold
        mask = scores >= self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return {"boxes": np.array([]), "scores": np.array([]), "class_ids": np.array([])}

        # Convert from xywh to xyxy format
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Remove padding and scale to original image size
        boxes_xyxy[:, [0, 2]] -= pad[0]  # x padding
        boxes_xyxy[:, [1, 3]] -= pad[1]  # y padding
        boxes_xyxy[:, [0, 2]] /= ratio[0]  # x ratio
        boxes_xyxy[:, [1, 3]] /= ratio[1]  # y ratio

        # Clip boxes to image boundaries
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_shape[1])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_shape[0])

        # Apply NMS
        keep_indices = nms_boxes(boxes_xyxy, scores, self.iou_thres)

        # Limit to max_det
        if len(keep_indices) > self.max_det:
            keep_indices = keep_indices[: self.max_det]

        return {
            "boxes": boxes_xyxy[keep_indices],
            "scores": scores[keep_indices],
            "class_ids": class_ids[keep_indices],
        }

    def _postprocess_obb(
        self,
        predictions: np.ndarray,
        ratio: tuple[float, float],
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """Postprocess for OBB (Oriented Bounding Box) task."""
        # Debug: Check predictions shape
        if predictions.shape[1] < 6:
            print(f"⚠️  Warning: Expected at least 6 features (4 for box + 1+ for classes + 1 for angle), got {predictions.shape}")
        
        # Extract boxes, scores, and angle
        # IMPORTANT: OBB ONNX format from Ultralytics is:
        # [x_center, y_center, width, height, class_score_0, class_score_1, ..., class_score_N, angle]
        # The angle is in the LAST column, NOT in column 4!
        boxes_xywh = predictions[:, :4]  # x_center, y_center, width, height
        scores_all = predictions[:, 4:-1]  # class scores (all columns except first 4 and last 1)
        angle = predictions[:, -1]  # angle in radians (last column)

        # Get max scores and class IDs
        scores = np.max(scores_all, axis=1)
        class_ids = np.argmax(scores_all, axis=1)
        
        print(f"🔍 Before filtering: {len(boxes_xywh)} predictions, max score: {scores.max():.3f}, min score: {scores.min():.3f}")
        print(f"🔍 Angle range: [{angle.min():.4f}, {angle.max():.4f}] radians ({angle.min()*180/np.pi:.1f}° to {angle.max()*180/np.pi:.1f}°)")

        # Filter by confidence threshold
        mask = scores >= self.conf_thres
        boxes_xywh = boxes_xywh[mask]
        angle = angle[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        print(f"🔍 After conf filter ({self.conf_thres}): {len(boxes_xywh)} detections")

        if len(boxes_xywh) == 0:
            return {"boxes": np.array([]), "scores": np.array([]), "class_ids": np.array([])}

        # Combine boxes with angle: [x_center, y_center, width, height, angle]
        boxes = np.concatenate([boxes_xywh, angle[:, None]], axis=1)

        # Debug: Show boxes before scaling
        if len(boxes) > 0:
            print(f"🔍 Before scaling - First box: center=({boxes[0,0]:.1f}, {boxes[0,1]:.1f}), size=({boxes[0,2]:.1f}×{boxes[0,3]:.1f}), angle={boxes[0,4]:.4f}")

        # Remove padding and scale to original image size
        boxes[:, 0] -= pad[0]  # x padding
        boxes[:, 1] -= pad[1]  # y padding
        boxes[:, 0] /= ratio[0]  # x ratio
        boxes[:, 1] /= ratio[1]  # y ratio
        boxes[:, 2] /= ratio[0]  # width ratio
        boxes[:, 3] /= ratio[1]  # height ratio
        # angle (column 4) remains unchanged

        # Debug: Show boxes after scaling
        if len(boxes) > 0:
            print(f"🔍 After scaling - First box: center=({boxes[0,0]:.1f}, {boxes[0,1]:.1f}), size=({boxes[0,2]:.1f}×{boxes[0,3]:.1f})")
            print(f"🔍 Image size: {orig_shape[1]}×{orig_shape[0]} (W×H)")

        # Clip centers to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_shape[0])

        # Debug: Show boxes after clipping
        if len(boxes) > 0:
            print(f"🔍 After clipping - First box: center=({boxes[0,0]:.1f}, {boxes[0,1]:.1f}), size=({boxes[0,2]:.1f}×{boxes[0,3]:.1f})")

        # Apply rotated NMS
        print(f"🔍 Before NMS: {len(boxes)} boxes, IoU threshold: {self.iou_thres}")
        keep_indices = nms_rotated(boxes, scores, self.iou_thres)
        print(f"🔍 After NMS: {len(keep_indices)} boxes kept")

        # Limit to max_det
        if len(keep_indices) > self.max_det:
            keep_indices = keep_indices[: self.max_det]

        return {
            "boxes": boxes[keep_indices],
            "scores": scores[keep_indices],
            "class_ids": class_ids[keep_indices],
        }

    def __call__(self, img: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run inference on an image.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            (dict): Dictionary containing boxes, scores, and class_ids.
        """
        # Preprocess
        img_input, ratio, pad, orig_shape = self.preprocess(img)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: img_input})

        # Postprocess
        results = self.postprocess(outputs, ratio, pad, orig_shape)

        return results

    def draw_detections(self, img: np.ndarray, results: dict[str, np.ndarray]) -> np.ndarray:
        """
        Draw detection results on image.

        Args:
            img (np.ndarray): Input image in BGR format.
            results (dict): Detection results from inference.

        Returns:
            (np.ndarray): Image with drawn detections.
        """
        img_draw = img.copy()

        boxes = results["boxes"]
        scores = results["scores"]
        class_ids = results["class_ids"]

        # Generate random colors for each class
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)

        if self.task == "obb":
            # Draw oriented bounding boxes using the same method as Ultralytics
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                x, y, w, h, angle = box
                color = tuple(int(c) for c in colors[class_id])

                # Convert xywhr to xyxyxyxy (4 corner points) like Ultralytics does
                # This matches the xywhr2xyxyxyxy function in ultralytics/utils/ops.py
                cos_value, sin_value = np.cos(angle), np.sin(angle)
                vec1 = np.array([w / 2 * cos_value, w / 2 * sin_value])
                vec2 = np.array([-h / 2 * sin_value, h / 2 * cos_value])
                ctr = np.array([x, y])
                
                # Calculate 4 corner points (clockwise from top-left)
                pt1 = ctr + vec1 + vec2  # top-right
                pt2 = ctr + vec1 - vec2  # bottom-right
                pt3 = ctr - vec1 - vec2  # bottom-left
                pt4 = ctr - vec1 + vec2  # top-left
                
                # Stack corner points into shape (4, 2)
                box_points = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)

                # Draw polygon using cv2.polylines (same as Ultralytics box_label with multi_points=True)
                cv2.polylines(img_draw, [box_points], True, color, thickness=2, lineType=cv2.LINE_AA)
                
                # Draw label
                label = f"{self.classes[class_id]}: {score:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_h += 3  # add padding like Ultralytics
                
                # Position label at first corner point (top-left-ish)
                p1 = (int(box_points[0][0]), int(box_points[0][1]))
                
                # Check if label fits outside box (above)
                outside = p1[1] >= label_h
                
                # Adjust if label extends beyond right edge
                if p1[0] > img_draw.shape[1] - label_w:
                    p1 = (img_draw.shape[1] - label_w, p1[1])
                
                # Calculate label background box
                p2 = (p1[0] + label_w, p1[1] - label_h if outside else p1[1] + label_h)
                
                # Draw label background
                cv2.rectangle(img_draw, p1, p2, color, -1, cv2.LINE_AA)
                
                # Draw label text
                cv2.putText(
                    img_draw,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + label_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
        else:
            # Draw regular bounding boxes
            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                color = tuple(int(c) for c in colors[class_id])

                # Draw bounding box
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label = f"{self.classes[class_id]}: {score:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_draw, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)

                # Draw label text
                cv2.putText(img_draw, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img_draw

    def crop_detections(
        self,
        img: np.ndarray,
        results: dict[str, np.ndarray],
        angle_type: int = 0,
    ) -> list[tuple[np.ndarray, dict[str, Any]]]:
        """
        Crop detected regions from image (works for OBB task).
        
        Args:
            img (np.ndarray): Input image in BGR format.
            results (dict): Detection results from inference.
            angle_type (int): Rotation direction in {0, 90, 180, 270} degrees (clockwise).
        
        Returns:
            (list): List of tuples (cropped_image, metadata) where metadata contains:
                - 'box': Original box coordinates
                - 'score': Confidence score
                - 'class_id': Class ID
                - 'class_name': Class name
                - 'transform': Transform matrices from crop_img_from_polygon
        """
        if self.task != "obb":
            print("⚠️  Warning: crop_detections is designed for OBB task, may not work correctly for regular detection")
            return []
        
        boxes = results["boxes"]
        scores = results["scores"]
        class_ids = results["class_ids"]
        
        cropped_results = []
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h, angle = box
            
            # Convert xywhr to 4 corner points (same as in draw_detections)
            cos_value, sin_value = np.cos(angle), np.sin(angle)
            vec1 = np.array([w / 2 * cos_value, w / 2 * sin_value])
            vec2 = np.array([-h / 2 * sin_value, h / 2 * cos_value])
            ctr = np.array([x, y])
            
            # Calculate 4 corner points
            pt1 = ctr + vec1 + vec2  # top-right
            pt2 = ctr + vec1 - vec2  # bottom-right
            pt3 = ctr - vec1 - vec2  # bottom-left
            pt4 = ctr - vec1 + vec2  # top-left
            
            polygon = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)
            
            # Crop the region
            try:
                cropped_img, transform_info = crop_img_from_polygon(img, polygon, angle_type)
                
                metadata = {
                    'box': box,
                    'score': score,
                    'class_id': class_id,
                    'class_name': self.classes[class_id],
                    'transform': transform_info,
                }
                
                cropped_results.append((cropped_img, metadata))
            except Exception as e:
                print(f"⚠️  Failed to crop detection {class_id}: {e}")
                continue
        
        return cropped_results


def main():
    """Main function to run ONNX inference."""
    parser = argparse.ArgumentParser(description="Standalone ONNX Inference for YOLO")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to output image")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections")
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "obb"], help="Task type: detect or obb")
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped detections (OBB only)")
    parser.add_argument("--crop-angle", type=int, default=0, choices=[0, 90, 180, 270], help="Rotation angle for crops")
    args = parser.parse_args()

    # Initialize inference
    model = ONNXInference(
        model_path=args.model,
        device=args.device,
        img_size=args.img_size,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        task=args.task,
    )

    # Read image
    img = cv2.imread(args.source)
    if img is None:
        raise ValueError(f"Could not read image from {args.source}")

    print(f"📸 Processing image: {args.source}")
    print(f"📐 Image shape: {img.shape}")

    # Run inference
    results = model(img)

    # Print results
    num_detections = len(results["boxes"])
    print(f"\n🎯 Found {num_detections} detections")

    if num_detections > 0:
        if args.task == "obb":
            for i, (box, score, class_id) in enumerate(
                zip(results["boxes"], results["scores"], results["class_ids"])
            ):
                x, y, w, h, angle = box
                angle_deg = angle * 180.0 / np.pi
                print(f"  {i + 1}. {model.classes[class_id]}: {score:.2f} at ({x:.1f}, {y:.1f}) size ({w:.1f}x{h:.1f}) angle {angle_deg:.1f}°")
        else:
            for i, (box, score, class_id) in enumerate(
                zip(results["boxes"], results["scores"], results["class_ids"])
            ):
                print(f"  {i + 1}. {model.classes[class_id]}: {score:.2f} at {box}")

    # Draw detections
    img_result = model.draw_detections(img, results)

    # Save result
    cv2.imwrite(args.output, img_result)
    print(f"\n💾 Saved result to: {args.output}")

    # Save cropped detections if requested
    if args.save_crops and args.task == "obb" and num_detections > 0:
        print(f"\n✂️  Cropping {num_detections} detections...")
        cropped_results = model.crop_detections(img, results, angle_type=args.crop_angle)
        
        # Create crops directory
        output_path = Path(args.output)
        crops_dir = output_path.parent / f"{output_path.stem}_crops"
        crops_dir.mkdir(exist_ok=True)
        
        for i, (cropped_img, metadata) in enumerate(cropped_results):
            class_name = metadata['class_name']
            score = metadata['score']
            crop_filename = crops_dir / f"crop_{i:03d}_{class_name}_{score:.2f}.jpg"
            cv2.imwrite(str(crop_filename), cropped_img)
            print(f"  💾 Saved crop {i+1}/{len(cropped_results)}: {crop_filename.name}")
        
        print(f"\n✅ Saved {len(cropped_results)} crops to: {crops_dir}")

    # Show result
    if args.show:
        cv2.imshow("ONNX Inference Result", img_result)
        print("\n👁️  Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
