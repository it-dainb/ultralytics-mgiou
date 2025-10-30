#!/usr/bin/env python3
"""
Debug classification loss - check dataset configuration.
"""

import yaml

def check_dataset_config(yaml_path):
    """Check dataset configuration for issues."""
    print("="*70)
    print("Checking Dataset Configuration")
    print("="*70)
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"\nDataset: {yaml_path}")
        print(f"\nConfiguration:")
        print(f"  Path: {config.get('path', 'NOT SET')}")
        print(f"  Train: {config.get('train', 'NOT SET')}")
        print(f"  Val: {config.get('val', 'NOT SET')}")
        print(f"  Number of classes (nc): {config.get('nc', 'NOT SET')}")
        
        names = config.get('names', [])
        print(f"\n  Class names ({len(names)} total):")
        if len(names) <= 20:
            for i, name in enumerate(names):
                print(f"    {i}: {name}")
        else:
            print(f"    Too many to display ({len(names)} classes)")
            print(f"    First 10: {names[:10]}")
            print(f"    ...")
            print(f"    ⚠️  WARNING: {len(names)} classes might be too many for your dataset size!")
        
        # Analysis
        print("\n" + "="*70)
        print("Analysis:")
        print("="*70)
        
        if len(names) > 80:
            print("  🔴 CRITICAL: Very high number of classes!")
            print(f"     With only 67 images, having {len(names)} classes will cause:")
            print("     - Severe class imbalance")
            print("     - High classification loss (exactly what we're seeing)")
            print("     - Poor learning")
            print("\n  💡 SOLUTION:")
            print("     1. Reduce number of classes (merge similar classes)")
            print("     2. Increase dataset size significantly")
            print(f"     3. Expect ~10-20 samples per class minimum ({len(names)*15}+ images)")
        
        elif len(names) > 20:
            print("  ⚠️  WARNING: Moderate-high number of classes")
            print(f"     With 67 images and {len(names)} classes:")
            print(f"     - Average ~{67/len(names):.1f} images per class")
            print("     - Might cause class imbalance issues")
        
        else:
            print(f"  ✅ Reasonable number of classes ({len(names)})")
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {yaml_path}")
        print("\nPlease provide the path to your dataset YAML file.")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("="*70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    else:
        # Try to guess common locations
        import os
        possible_paths = [
            'data.yaml',
            'dataset.yaml',
            'polygon.yaml',
            '../data.yaml',
        ]
        
        yaml_path = None
        for path in possible_paths:
            if os.path.exists(path):
                yaml_path = path
                break
        
        if yaml_path is None:
            print("Usage: python debug_cls_loss.py <path_to_dataset.yaml>")
            print("\nOr place your dataset YAML in one of:")
            for p in possible_paths:
                print(f"  - {p}")
            sys.exit(1)
    
    check_dataset_config(yaml_path)
