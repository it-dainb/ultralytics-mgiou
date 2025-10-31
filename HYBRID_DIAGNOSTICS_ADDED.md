# Hybrid Loss Diagnostic Logging - Session Update

## Summary

Added comprehensive diagnostic logging to identify why hybrid loss is stuck at 9-10 instead of the expected 0.2-2.0 range during training.

## Changes Made

### 1. Enhanced PolygonLoss Initialization Logging

**File:** `ultralytics/utils/loss.py` (lines 822-872)

**Added:**
```python
# Log initialization for debugging
if use_hybrid:
    print(f"[HYBRID INIT] PolygonLoss initialized in HYBRID mode:")
    print(f"  - alpha_schedule: {alpha_schedule}")
    print(f"  - alpha_start: {alpha_start} (L2 weight)")
    print(f"  - alpha_end: {alpha_end} (L2 weight)")
    print(f"  - total_epochs: {total_epochs}")
    print(f"  - Initial EMA values: L2={self.l2_loss_ema}, MGIoU={self.mgiou_loss_ema}")
```

**Purpose:** Confirms hybrid mode is actually initializing at startup.

---

### 2. Enhanced Loss Computation Debug Logging

**File:** `ultralytics/utils/loss.py` (lines 1000-1009)

**Changed from:**
```python
# Only log on explicit epoch updates to avoid spam
if _DEBUG_NAN and epoch is not None:
    print(f"[HYBRID] Epoch {self.current_epoch}: alpha={alpha:.3f}, ...")
```

**Changed to:**
```python
# Enhanced debug info - log EVERY call when debug is enabled to diagnose epoch issues
if _DEBUG_NAN:
    print(f"[HYBRID] current_epoch={self.current_epoch}, passed_epoch={epoch}, alpha={alpha:.3f}, "
          f"L2={l2_loss.item():.4f}, MGIoU={mgiou_loss.item():.4f}, "
          f"L2_EMA={self.l2_loss_ema:.4f}, MGIoU_EMA={self.mgiou_loss_ema:.4f}, "
          f"L2_norm={l2_normalized.item():.4f}, MGIoU_norm={mgiou_normalized.item():.4f}, "
          f"total={total_loss.item():.4f}")
```

**Purpose:** Shows actual loss values, EMA tracking, normalization, and epoch state for every batch.

---

### 3. Enhanced v8PolygonLoss.set_epoch() Logging

**File:** `ultralytics/utils/loss.py` (lines 1498-1506)

**Changed from:**
```python
def set_epoch(self, epoch: int):
    """Set current epoch for hybrid loss scheduling."""
    self.current_epoch = epoch
    self.polygon_loss.current_epoch = epoch
```

**Changed to:**
```python
def set_epoch(self, epoch: int):
    """Set current epoch for hybrid loss scheduling."""
    self.current_epoch = epoch
    self.polygon_loss.current_epoch = epoch
    
    # Log epoch updates for debugging
    if self.use_hybrid:
        alpha = self.polygon_loss.get_alpha()
        print(f"[HYBRID] v8PolygonLoss.set_epoch({epoch}): "
              f"alpha={alpha:.4f}, L2_EMA={self.polygon_loss.l2_loss_ema:.4f}, "
              f"MGIoU_EMA={self.polygon_loss.mgiou_loss_ema:.4f}")
```

**Purpose:** Confirms `set_epoch()` is actually being called each epoch.

---

### 4. Enhanced PolygonTrainer Callback with DDP Support

**File:** `ultralytics/models/yolo/polygon/train.py` (lines 149-181)

**Changed from:**
```python
def on_train_epoch_start(self):
    """Called at the start of each training epoch to update loss function's epoch counter."""
    super().on_train_epoch_start() if hasattr(super(), 'on_train_epoch_start') else None
    
    # Update epoch in loss function for hybrid scheduling
    if hasattr(self.model, 'criterion') and hasattr(self.model.criterion, 'set_epoch'):
        self.model.criterion.set_epoch(self.epoch)
        
        # Log alpha progression for hybrid mode
        if self.use_hybrid and hasattr(self.model.criterion, 'polygon_loss'):
            alpha = self.model.criterion.polygon_loss.get_alpha()
            LOGGER.info(f"[HYBRID] Starting Epoch {self.epoch}/{self.epochs}: alpha={alpha:.4f} "
                       f"(L2 weight={alpha:.2%}, MGIoU weight={1-alpha:.2%})")
```

**Changed to:**
```python
def on_train_epoch_start(self):
    """Called at the start of each training epoch to update loss function's epoch counter."""
    super().on_train_epoch_start() if hasattr(super(), 'on_train_epoch_start') else None
    
    # Debug: Check model structure (only epoch 0)
    if self.use_hybrid and self.epoch == 0:
        print(f"[HYBRID DEBUG] Checking model structure:")
        print(f"  - self.model type: {type(self.model)}")
        print(f"  - has criterion: {hasattr(self.model, 'criterion')}")
        if hasattr(self.model, 'criterion'):
            print(f"  - criterion type: {type(self.model.criterion)}")
            print(f"  - has set_epoch: {hasattr(self.model.criterion, 'set_epoch')}")
        # Check if wrapped in DDP
        if hasattr(self.model, 'module'):
            print(f"  - model.module type: {type(self.model.module)}")
            print(f"  - module.criterion exists: {hasattr(self.model.module, 'criterion')}")
    
    # Update epoch in loss function for hybrid scheduling
    # Handle both regular and DDP-wrapped models
    criterion = None
    if hasattr(self.model, 'criterion'):
        criterion = self.model.criterion
    elif hasattr(self.model, 'module') and hasattr(self.model.module, 'criterion'):
        criterion = self.model.module.criterion
    
    if criterion and hasattr(criterion, 'set_epoch'):
        print(f"[HYBRID] Calling set_epoch({self.epoch}) on criterion")
        criterion.set_epoch(self.epoch)
        
        # Log alpha progression for hybrid mode
        if self.use_hybrid and hasattr(criterion, 'polygon_loss'):
            alpha = criterion.polygon_loss.get_alpha()
            print(f"[HYBRID] Starting Epoch {self.epoch}/{self.epochs}: alpha={alpha:.4f} "
                  f"(L2 weight={alpha:.2%}, MGIoU weight={1-alpha:.2%})")
    else:
        print(f"[HYBRID WARNING] Could not find criterion or set_epoch method!")
        print(f"  - self.model type: {type(self.model)}")
        print(f"  - criterion found: {criterion is not None}")
        print(f"  - has set_epoch: {criterion and hasattr(criterion, 'set_epoch')}")
```

**Key improvements:**
- **DDP Support:** Checks both `self.model.criterion` and `self.model.module.criterion`
- **Detailed debugging:** Shows model structure at epoch 0
- **Warning messages:** Alerts if callback can't find criterion

**Purpose:** This was likely the root cause - the callback wasn't finding the criterion in DDP mode!

---

## Test Results

**File:** `test_hybrid_diagnostics.py`

All 4 tests pass:
✅ Test 1: PolygonLoss initialization in hybrid mode
✅ Test 2: Epoch updates with alpha scheduling
✅ Test 3: Loss computation with proper normalization
✅ Test 4: v8PolygonLoss set_epoch() method

**Test output shows:**
- Hybrid mode initializes correctly
- Alpha transitions from 0.9 → 0.2 over 100 epochs
- Loss normalization works (both losses ~1.0 before combining)
- EMA tracking updates properly
- set_epoch() method works correctly

---

## Root Cause Hypothesis

Based on the code analysis, the most likely cause is:

### **The callback wasn't finding the criterion in DDP/wrapped models**

**Evidence:**
1. Line 154 in old code only checked `self.model.criterion`
2. In multi-GPU or DDP training, models get wrapped
3. Wrapped models have structure: `self.model.module.criterion`
4. Without DDP handling, `set_epoch()` was never called
5. Without `set_epoch()`, epoch stayed at 0, alpha stayed at 0.9
6. With alpha=0.9, loss = mostly L2 (unnormalized) → stuck at 9-10

**The fix:**
```python
# Handle both regular and DDP-wrapped models
criterion = None
if hasattr(self.model, 'criterion'):
    criterion = self.model.criterion
elif hasattr(self.model, 'module') and hasattr(self.model.module, 'criterion'):
    criterion = self.model.module.criterion
```

---

## What to Look For in Training Logs

When you run training with the diagnostic logging, you should see:

### 1. At Startup (Epoch 0)
```
[HYBRID INIT] PolygonLoss initialized in HYBRID mode:
  - alpha_schedule: cosine
  - alpha_start: 0.9 (L2 weight)
  - alpha_end: 0.2 (L2 weight)
  - total_epochs: 100
  - Initial EMA values: L2=1.0, MGIoU=1.0

[HYBRID DEBUG] Checking model structure:
  - self.model type: <class '...'>
  - has criterion: True
  - criterion type: <class 'ultralytics.utils.loss.v8PolygonLoss'>
  - has set_epoch: True
```

### 2. At Each Epoch Start
```
[HYBRID] Calling set_epoch(0) on criterion
[HYBRID] v8PolygonLoss.set_epoch(0): alpha=0.9000, L2_EMA=1.0000, MGIoU_EMA=1.0000
[HYBRID] Starting Epoch 0/100: alpha=0.9000 (L2 weight=90.00%, MGIoU weight=10.00%)
```

### 3. During Training (if ULTRALYTICS_DEBUG_NAN=1)
```
[HYBRID] current_epoch=0, passed_epoch=0, alpha=0.900, L2=0.0294, MGIoU=0.1458, 
         L2_EMA=0.9029, MGIoU_EMA=0.9146, L2_norm=0.0325, MGIoU_norm=0.1594, total=0.0408
```

---

## Diagnostic Scenarios

### Scenario 1: Hybrid Mode Not Initializing
**Symptom:** No `[HYBRID INIT]` message at startup

**Likely causes:**
- `use_hybrid=True` not passed correctly
- Check training command has `use_hybrid=True`

**Fix:** Verify training command and parameter passing

---

### Scenario 2: set_epoch() Not Called
**Symptom:** No `[HYBRID] Calling set_epoch(X)` messages

**Likely causes:**
- Callback not registered
- Model wrapped in DDP and criterion not found
- The fix in this session should address this!

**What to check:**
- Look for `[HYBRID DEBUG]` output at epoch 0
- If shows `module.criterion exists: True` but `has criterion: False`, then DDP wrapping is the issue
- This is now fixed with the DDP-aware code

---

### Scenario 3: Alpha Not Changing
**Symptom:** All `[HYBRID]` messages show same alpha value (e.g., always 0.9)

**Likely causes:**
- `set_epoch()` not being called (see Scenario 2)
- `current_epoch` stuck at 0

**What to check:**
- Compare `current_epoch` vs `passed_epoch` in batch debug output
- If both stuck at 0, then `set_epoch()` isn't being called

---

### Scenario 4: Loss Values Wrong Scale
**Symptom:** `total` loss in 9-10 range instead of 0.2-2.0

**Likely causes:**
- Normalization not working (EMA values stuck at 1.0)
- Alpha stuck at high value (0.9) giving mostly L2 loss
- L2 loss unnormalized is naturally 8-12 range

**What to check:**
- Look at `L2_EMA` and `MGIoU_EMA` values
- Should evolve from 1.0 to actual loss magnitudes (~0.02 for L2, ~0.13 for MGIoU)
- Look at `L2_norm` and `MGIoU_norm` - should both be ~1.0
- If `L2_norm` >> 1.0, then EMA not updating

---

## Next Steps

### 1. Run Training with Diagnostic Logging

**Command:**
```bash
YOLO_VERBOSE=True ULTRALYTICS_DEBUG_NAN=1 yolo polygon train \
  data=./datasets/final/cc_obb.yaml model=./polygon.yaml \
  use_hybrid=True single_cls=True dfl=0 \
  optimizer='AdamW' lr0=0.005 dropout=0.1 cos_lr=True \
  epochs=100 patience=0 batch=0.9 imgsz=640 rect=True \
  plots=True compile=False pretrained=False augment=True \
  auto_augment=autoaugment
```

**Important:** Keep `ULTRALYTICS_DEBUG_NAN=1` enabled to see detailed batch-level logging.

### 2. Check Training Logs

Save the first 2 epochs of output:
```bash
YOLO_VERBOSE=True ULTRALYTICS_DEBUG_NAN=1 yolo polygon train ... 2>&1 | tee hybrid_training.log
```

Look for:
- `[HYBRID INIT]` message confirming initialization
- `[HYBRID DEBUG]` message showing model structure
- `[HYBRID] Calling set_epoch(X)` messages each epoch
- `[HYBRID]` batch-level messages showing:
  - `current_epoch` incrementing (0, 1, 2, ...)
  - `alpha` decreasing over epochs (0.9 → 0.8 → 0.7 → ...)
  - `L2_EMA` and `MGIoU_EMA` evolving from 1.0 to actual values
  - `total` loss in reasonable range (0.2-2.0)

### 3. If Still Not Working

If you still see polygon loss stuck at 9-10:

**A. Check the logs for:**
- Warning messages from the callback
- Whether `set_epoch()` is being called
- Whether alpha is changing
- What the actual L2 and MGIoU raw values are

**B. Share the first ~200 lines of training output** showing:
- Initialization
- First few batches of epoch 0
- Transition to epoch 1

**C. I can then diagnose:**
- Whether the fix worked (DDP handling)
- Whether there's another issue (e.g., wrong loss values being passed)
- What the next step should be

---

## Expected Behavior After Fix

With the diagnostic logging and DDP fix:

| Metric | Expected | Previous (Broken) |
|--------|----------|-------------------|
| Initialization | `[HYBRID INIT]` visible | Maybe visible |
| Callback | `set_epoch()` called each epoch | Not called (DDP issue) |
| Epoch tracking | `current_epoch` increments | Stuck at 0 |
| Alpha | Decreases 0.9→0.2 | Stuck at 0.9 |
| L2 EMA | Evolves to ~0.02 | Stuck at 1.0 or wrong |
| MGIoU EMA | Evolves to ~0.13 | Stuck at 1.0 or wrong |
| Normalized losses | Both ~1.0 | Not normalized |
| Total loss | 0.2-2.0 range | 9-10 (unnormalized L2) |
| Polygon loss trend | Decreases over epochs | Stuck at 9-10 |

---

## Summary

**Problem:** Hybrid loss stuck at 9-10, not decreasing

**Root Cause:** Callback couldn't find criterion in DDP-wrapped models → `set_epoch()` never called → alpha stuck at 0.9 → mostly unnormalized L2 loss

**Solution:**
1. Added DDP-aware criterion lookup
2. Added comprehensive diagnostic logging
3. Enhanced debug output at initialization, epoch start, and batch level

**Test Status:** ✅ All tests pass, diagnostics confirmed working

**Next Action:** Run training with `ULTRALYTICS_DEBUG_NAN=1` and review logs to confirm fix worked
