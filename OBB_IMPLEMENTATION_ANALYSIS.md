# OBB Implementation Analysis: Old vs Working Code

**Date:** November 7, 2025  
**Purpose:** Document why the old OBB implementation failed and how the working implementation fixes the issues

---

## 🔴 Critical Issue #1: Incorrect ONNX Output Format Interpretation

### Old Code (WRONG) ❌

```python
# In _utils.py - xywh_angle_to_corners function
boxes = np.asarray(boxes)
x, y, w, h = boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3] 
ang = boxes[0, -1]  # ❌ Getting angle from WRONG position
```

### Working Code (CORRECT) ✅

```python
# In onnx_infer.py - _postprocess_obb function
boxes_xywh = predictions[:, :4]  # x_center, y_center, width, height
scores_all = predictions[:, 4:-1]  # ✅ class scores (all columns except first 4 and last 1)
angle = predictions[:, -1]  # ✅ angle in radians (LAST column)
```

### Why it Failed

- **Root Cause:** Misunderstanding of ONNX output tensor structure
- **ONNX Output Format:** `[x, y, w, h, class_0, class_1, ..., class_N, angle]`
- **Old code assumption:** Angle is at a fixed position (column 4)
- **Reality:** Angle is always in the LAST column, after ALL class scores
- **Impact:** Extracted angle from wrong position → incorrect rotated boxes → wrong NMS → bad detections

### Key Learning

```python
# IMPORTANT: OBB ONNX format from Ultralytics is:
# [x_center, y_center, width, height, class_score_0, ..., class_score_N, angle]
# The angle is in the LAST column, NOT in column 4!
```

The number of class score columns is variable (depends on the dataset), so you CANNOT use hardcoded indices.

---

## 🔴 Critical Issue #2: Coordinate System Confusion

### Old Code (WRONG) ❌

```python
# In _utils.py - obb_postprocess
box_points, angle_deg = xywh_angle_to_corners(det, normalized=False)

# Then scaling:
h, w = orig_img.shape[:2]
scale_x = w / 640
scale_y = h / 640

boxes = np.array(boxes, dtype=np.float32)
boxes[..., 0] *= scale_x  # x
boxes[..., 1] *= scale_y  # y
```

### Working Code (CORRECT) ✅

```python
# In onnx_infer.py - _postprocess_obb
# Step 1: Remove padding FIRST
boxes[:, 0] -= pad[0]  # x padding
boxes[:, 1] -= pad[1]  # y padding

# Step 2: Then scale to original size
boxes[:, 0] /= ratio[0]  # x ratio
boxes[:, 1] /= ratio[1]  # y ratio
boxes[:, 2] /= ratio[0]  # width ratio
boxes[:, 3] /= ratio[1]  # height ratio

# Step 3: Clip centers to image boundaries
boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_shape[1])
boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_shape[0])
```

### Why it Failed

1. **Wrong Order:** You scaled before removing padding
   - Correct order: Remove padding → Scale → Clip
   - Your order: Scale → (padding never removed properly)

2. **Missing Transformations:**
   - Forgot to scale width and height (only scaled x, y)
   - Didn't account for letterbox padding offsets

3. **Coordinate Space Mismatch:**
   - Predictions are in letterboxed image space (640x640)
   - Need to transform to original image space
   - Your transformation was incomplete

### Mathematical Explanation

```
Letterbox Preprocessing:
original_image (H, W) → resize → (H', W') → add_padding → (640, 640)

Correct Inverse Transform:
(640, 640) → remove_padding → (H', W') → scale → (H, W)

Old (Wrong) Transform:
(640, 640) → scale → ??? (mixed coordinate system)
```

---

## 🔴 Critical Issue #3: NMS Applied on Wrong Format

### Old Code (WRONG) ❌

```python
# In _utils.py - non_max_suppression
if rotated:
    boxes = np.concatenate((x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1)  # xywhr
    # ❌ But x[:, -1:] is NOT the angle! It's mask data or something else
    i = nms_rotated(boxes, scores, iou_thres)
```

### Working Code (CORRECT) ✅

```python
# In onnx_infer.py - _postprocess_obb

# Step 1: Extract angle correctly from raw predictions
angle = predictions[:, -1]  # Last column from ONNX output

# Step 2: Filter by confidence
mask = scores >= self.conf_thres
boxes_xywh = boxes_xywh[mask]
angle = angle[mask]
scores = scores[mask]
class_ids = class_ids[mask]

# Step 3: Combine boxes with angle properly
boxes = np.concatenate([boxes_xywh, angle[:, None]], axis=1)

# Step 4: Apply NMS with correct format
keep_indices = nms_rotated(boxes, scores, self.iou_thres)
```

### Why it Failed

- **Wrong Angle Values:** `x[:, -1:]` was NOT the angle column
  - At that point in your pipeline, `x` had format: `[box_coords, conf, cls, mask_coeffs]`
  - Angle was lost earlier in the processing

- **Corrupted IoU Calculations:**
  - `batch_probiou()` uses angle to calculate rotated IoU
  - Wrong angles → wrong IoU → NMS keeps/removes wrong boxes

- **Pipeline Design Flaw:**
  - You transformed data multiple times, losing track of what each column represented
  - Working code keeps clear separation: extract → filter → combine → NMS

---

## 🔴 Critical Issue #4: Preprocessing Inconsistency

### Old Code (WRONG) ❌

```python
# In _utils.py - LetterBox
if shape[::-1] != new_unpad:  # resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

# ❌ RESIZE AGAIN after padding!
img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
```

### Working Code (CORRECT) ✅

```python
# In onnx_infer.py - LetterBox
if shape[::-1] != new_unpad:  # resize
    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

# Add padding (no second resize)
image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
```

### Why it Failed

1. **Double Resizing:** You resized TWICE
   - First resize: preserve aspect ratio
   - Add padding: make it square
   - Second resize: **DISTORTS everything** ❌

2. **Invalid Padding/Ratio:**
   - Your `ratio` and `pad` values were calculated for the FIRST resize
   - After second resize, these values are meaningless
   - Postprocessing used wrong values → incorrect coordinate transformation

3. **Model Expectations:**
   - YOLO models expect letterbox preprocessing (resize + pad)
   - Your preprocessing created distorted images
   - Model outputs were based on distorted input → unreliable

### Visual Explanation

```
Correct Letterbox:
[Original 800x600] → resize → [640x480] → pad → [640x640]
                       ↑                    ↑
                   aspect ratio         centered
                   preserved            padding

Wrong (Old Code):
[Original 800x600] → resize → [640x480] → pad → [640x640] → resize → [640x640]
                                                                ↑
                                                          DISTORTION!
                                                      (squashes the image)
```

---

## 🔴 Critical Issue #5: Missing Debug Information

### Old Code ❌

```python
# No debug prints
detections = non_max_suppression(prediction=preds, ...)
if detections is None or len(detections) == 0:
    return None, None, None
# User has NO IDEA what went wrong
```

### Working Code ✅

```python
print(f"🔍 Before filtering: {len(boxes_xywh)} predictions, max score: {scores.max():.3f}")
print(f"🔍 Angle range: [{angle.min():.4f}, {angle.max():.4f}] radians")
print(f"🔍 After conf filter ({self.conf_thres}): {len(boxes_xywh)} detections")
print(f"🔍 Before scaling - First box: center=({boxes[0,0]:.1f}, {boxes[0,1]:.1f})")
print(f"🔍 After scaling - First box: center=({boxes[0,0]:.1f}, {boxes[0,1]:.1f})")
print(f"🔍 Before NMS: {len(boxes)} boxes, IoU threshold: {self.iou_thres}")
print(f"🔍 After NMS: {len(keep_indices)} boxes kept")
```

### Why it Matters

- **Debugging Visibility:** Can trace exactly where the pipeline breaks
- **Data Validation:** Can verify angle ranges, box coordinates, score distributions
- **Performance Analysis:** Can identify bottlenecks (too many predictions, NMS too strict, etc.)
- **User Feedback:** Users know what's happening instead of silent failures

---

## 📊 Summary Comparison Table

| Aspect | Old Code ❌ | Working Code ✅ |
|--------|------------|----------------|
| **Angle extraction** | `predictions[:, -1]` after data mixing | `predictions[:, -1]` from raw ONNX output |
| **Number of classes** | Hardcoded assumptions | Dynamic handling with `scores_all = predictions[:, 4:-1]` |
| **Coordinate transform order** | Scale → Remove padding (wrong) | Remove padding → Scale → Clip (correct) |
| **Width/height scaling** | Missing (only scaled x, y) | Properly scaled all 4 dimensions |
| **Preprocessing** | Double resize (distortion) | Single resize + padding (letterbox) |
| **NMS input format** | Wrong format with incorrect angles | Correct `[x, y, w, h, angle]` format |
| **Debug visibility** | No logging | Extensive debug prints at each stage |
| **Error handling** | Silent failures | Verbose error messages |
| **Code documentation** | Minimal comments | Detailed comments explaining ONNX format |

---

## 🎯 Root Cause Analysis

### The Fundamental Problem

**Misunderstanding the ONNX output tensor structure** for OBB models.

```python
# What you assumed:
# [x, y, w, h, angle, class_scores...]  ❌

# Reality (Ultralytics OBB ONNX format):
# [x, y, w, h, class_score_0, class_score_1, ..., class_score_N, angle]  ✅
```

### Cascade of Failures

```
Wrong angle extraction
    ↓
Wrong box format for NMS
    ↓
Wrong IoU calculations
    ↓
Wrong boxes kept/removed
    ↓
Wrong coordinate transformations (no padding removal)
    ↓
Boxes in wrong coordinate system
    ↓
Failed detections
```

---

## 🛠️ Key Lessons Learned

1. **Always Read Model Output Documentation**
   - Don't assume output format
   - Check model export code or documentation
   - Validate with debug prints

2. **Coordinate System Transformations Must Be Precise**
   - Order matters: Padding removal before scaling
   - All dimensions must be transformed (x, y, w, h)
   - Keep track of coordinate spaces at each step

3. **Preprocessing Must Match Model Expectations**
   - YOLO expects letterbox (resize + pad, NOT double resize)
   - Any deviation breaks the model's learned representations

4. **Debug Information is Critical**
   - Add prints at each pipeline stage
   - Validate data ranges and shapes
   - Makes debugging 10x faster

5. **Keep Data Format Consistent**
   - Define clear formats (e.g., `xywhr` for rotated boxes)
   - Don't mix formats in the same array
   - Document what each column represents

---

## 📝 Code Architecture Comparison

### Old Code Architecture (Fragmented)

```
Image → LetterBox (buggy) → ONNX Model → non_max_suppression (complex, wrong angle)
                                              ↓
                                         xywh_angle_to_corners (wrong input)
                                              ↓
                                         Scale (wrong order)
                                              ↓
                                         Draw (wrong coordinates)
```

### Working Code Architecture (Clean)

```
Image → LetterBox (correct) → ONNX Model → _postprocess_obb (OBB-specific)
                                              ↓
                               [Extract angle from last column]
                                              ↓
                               [Filter by confidence]
                                              ↓
                               [Remove padding]
                                              ↓
                               [Scale to original size]
                                              ↓
                               [Apply rotated NMS]
                                              ↓
                               Draw (correct coordinates)
```

---

## ✅ Recommended Fixes for Old Code

If you want to fix the old implementation:

### Fix 1: Extract Angle Correctly

```python
# Before NMS, extract angle from raw predictions
angle = predictions[:, -1]  # LAST column
scores_all = predictions[:, 4:-1]  # All class scores
```

### Fix 2: Fix Coordinate Transform Order

```python
# Remove padding first
boxes[:, 0] -= pad[0]
boxes[:, 1] -= pad[1]

# Then scale
boxes[:, :4] /= ratio

# Then clip
boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_shape[1])
boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_shape[0])
```

### Fix 3: Fix LetterBox (Remove Double Resize)

```python
# Remove this line:
# img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
```

### Fix 4: Add Debug Prints

```python
print(f"🔍 Predictions shape: {predictions.shape}")
print(f"🔍 Angle range: [{angle.min():.4f}, {angle.max():.4f}]")
print(f"🔍 After NMS: {len(keep_indices)} detections")
```

---

## 🎓 Conclusion

The old implementation failed due to a **fundamental misunderstanding of the ONNX OBB output format**, compounded by:
- Incorrect coordinate transformations
- Buggy preprocessing (double resize)
- Wrong angle extraction
- Lack of debugging information

The working implementation succeeds by:
- Correctly interpreting ONNX output (`angle` is LAST column)
- Proper coordinate system transformations (padding → scaling → clipping)
- Clean preprocessing (single resize + padding)
- Extensive debugging and validation
- Clear code architecture with OBB-specific postprocessing

**The most important takeaway:** Always validate your understanding of model outputs with debug prints before building complex pipelines on top of them.
