"""Diagnostic: Check which version of loss.py is being used."""
import sys
import inspect

print("="*80)
print("DIAGNOSTIC: Checking loss.py version")
print("="*80)

# Import without triggering local package
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
try:
    import ultralytics.utils.loss as loss_module
    
    print(f"\nFile location: {loss_module.__file__}")
    print(f"\n_EPS value: {loss_module._EPS}")
    print(f"_DEBUG_NAN: {loss_module._DEBUG_NAN}")
    
    # Check if our safety fixes are present
    source = inspect.getsource(loss_module.MGIoUPoly.forward)
    
    checks = {
        "Layer 1: Projection NaN/Inf replacement": "torch.isnan(proj1) | torch.isinf(proj1)" in source,
        "Layer 2: Inter NaN replacement": "torch.where(torch.isnan(inter)" in source,
        "Layer 2: Hull NaN replacement": "torch.where(torch.isnan(hull)" in source,
        "Layer 3: iou_term NaN replacement": "torch.where(torch.isnan(iou_term)" in source,
        "Layer 3: penalty_term NaN replacement": "torch.where(torch.isnan(penalty_term)" in source,
        "Layer 3: GIoU1D NaN replacement (fast_mode)": "torch.where(torch.isnan(giou1d)" in source,
        "Enhanced debug with extra_info": "extra_info" in source,
    }
    
    print("\n" + "="*80)
    print("Safety mechanisms present:")
    print("="*80)
    for name, present in checks.items():
        status = "✓ PRESENT" if present else "✗ MISSING"
        print(f"{status}: {name}")
    
    if all(checks.values()):
        print("\n✓ All safety fixes are present!")
        print("\nYou can now run training with:")
        print("  export ULTRALYTICS_DEBUG_NAN=1")
        print("  python train.py")
    else:
        print("\n✗ Some safety fixes are MISSING!")
        print("\nAction required: Install local package")
        print("  pip uninstall ultralytics -y")
        print("  pip install -e /home/dainb_1@digi-texx.local/PROJECTS_locals/ultralytics-mgiou")
        
except Exception as e:
    print(f"\nError: {e}")
    print("\nTrying to check if local modifications are correct...")
    sys.path.insert(0, '/home/dainb_1@digi-texx.local/PROJECTS_locals/ultralytics-mgiou')
    with open('/home/dainb_1@digi-texx.local/PROJECTS_locals/ultralytics-mgiou/ultralytics/utils/loss.py', 'r') as f:
        local_source = f.read()
    
    checks_local = {
        "Layer 3: iou_term NaN fix": "torch.where(torch.isnan(iou_term)" in local_source,
        "Layer 3: penalty_term NaN fix": "torch.where(torch.isnan(penalty_term)" in local_source,
    }
    
    print("\nLocal file checks:")
    for name, present in checks_local.items():
        status = "✓ PRESENT" if present else "✗ MISSING"
        print(f"{status}: {name}")
    
    if all(checks_local.values()):
        print("\n✓ Local fixes look good, need to install the package")
