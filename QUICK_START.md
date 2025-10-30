# Quick Start Guide - NaN Fix Installation

## Problem
Training crashes with NaN errors in GIoU computation after division operations.

## Solution Status
✓ All fixes implemented in local code  
⚠ Need to install to system Python

---

## Installation (Choose One Method)

### Method 1: Automated Script (Recommended)
```bash
cd /home/dainb_1@digi-texx.local/PROJECTS_locals/ultralytics-mgiou
bash install_fixes.sh
```

### Method 2: Manual Installation
```bash
# Find which Python has torch
python3 -c "import torch; print('OK')"  # or python3.12

# Uninstall old version (may need sudo)
sudo pip uninstall ultralytics -y

# Install local version
sudo pip install -e /home/dainb_1@digi-texx.local/PROJECTS_locals/ultralytics-mgiou
```

### Method 3: Create New Conda Environment
```bash
conda create -n ultralytics python=3.12 pytorch torchvision opencv -c pytorch -y
conda activate ultralytics
pip install -e /home/dainb_1@digi-texx.local/PROJECTS_locals/ultralytics-mgiou
```

---

## Verification

After installation, run:
```bash
python3 check_version.py
```

Expected output:
```
✓ PRESENT: Layer 3: iou_term NaN replacement
✓ PRESENT: Layer 3: penalty_term NaN replacement
✓ All safety fixes are present!
```

---

## Run Training

```bash
export ULTRALYTICS_DEBUG_NAN=1
python3 train.py  # or your training script
```

---

## What Was Fixed

**Layer 3 (Critical):** Added NaN replacement immediately after division operations
- Location: `ultralytics/utils/loss.py:470-473`
- Catches NaN from `inter / union_safe` and `(hull - union) / hull`
- Replaces NaN with 0 (zero contribution from degenerate axes)

**Why needed:** Even with epsilon clamping, divisions can produce NaN due to floating-point precision limits when both numerator and denominator are near machine epsilon.

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
→ Wrong Python environment. Install torch or use correct environment.

**"Permission denied"**
→ Add `sudo` before pip commands if ultralytics is installed system-wide.

**"All fixes present but still getting NaN"**
→ Check that training script uses the same Python:
```bash
which python3
python3 -c "import ultralytics; print(ultralytics.__file__)"
```

---

## Files Changed
- `ultralytics/utils/loss.py` - Core fixes
- `check_version.py` - Diagnostic tool
- `install_fixes.sh` - Auto-installer
- `RESUME_STATUS.md` - Detailed status
- `QUICK_START.md` - This file

---

## Support

If issues persist after installation:
1. Run `python3 check_version.py` and share output
2. Enable debug: `export ULTRALYTICS_DEBUG_NAN=1`
3. Share training error output
