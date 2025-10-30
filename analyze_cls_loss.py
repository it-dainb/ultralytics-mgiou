#!/usr/bin/env python3
"""
Analyze why cls_loss is so high with nc=1.
Check the actual tensor shapes and values.
"""

import torch
import torch.nn as nn

def simulate_yolo_cls_loss():
    """Simulate YOLO classification loss computation."""
    print("="*70)
    print("Simulating YOLO Classification Loss (nc=1)")
    print("="*70)
    
    # Typical YOLO configuration
    batch_size = 8
    num_anchors = 8400  # P3, P4, P5 combined
    nc = 1  # Your dataset
    
    # Simulate predictions (logits before sigmoid)
    pred_scores = torch.randn(batch_size, num_anchors, nc)
    
    # Target scores from assigner (after Task-Aligned matching)
    # Most anchors get score 0 (background), few get >0 (matched to objects)
    target_scores = torch.zeros(batch_size, num_anchors, nc)
    
    # Simulate ~15 matched anchors per image (typical for small objects)
    instances_per_image = 15
    for b in range(batch_size):
        matched_indices = torch.randperm(num_anchors)[:instances_per_image]
        target_scores[b, matched_indices, 0] = torch.rand(instances_per_image) * 0.8 + 0.2  # 0.2-1.0
    
    # Compute loss the YOLO way
    bce = nn.BCEWithLogitsLoss(reduction="none")
    bce_loss = bce(pred_scores, target_scores)  # Shape: [batch, anchors, nc]
    
    print(f"\nTensor shapes:")
    print(f"  pred_scores: {pred_scores.shape}")
    print(f"  target_scores: {target_scores.shape}")
    print(f"  bce_loss: {bce_loss.shape}")
    
    print(f"\nTarget statistics:")
    print(f"  Positive anchors: {(target_scores > 0).sum().item()}")
    print(f"  Background anchors: {(target_scores == 0).sum().item()}")
    print(f"  target_scores sum: {target_scores.sum().item():.2f}")
    
    # YOLO formula
    target_scores_sum = max(target_scores.sum(), 1)
    cls_loss = bce_loss.sum() / target_scores_sum
    
    print(f"\nLoss computation:")
    print(f"  bce_loss.sum(): {bce_loss.sum().item():.2f}")
    print(f"  target_scores_sum: {target_scores_sum.item():.2f}")
    print(f"  cls_loss (before hyp.cls): {cls_loss.item():.2f}")
    
    # Apply hyperparameter weight (default cls=0.5)
    hyp_cls = 0.5
    cls_loss_weighted = cls_loss * hyp_cls * batch_size  # YOLO multiplies by batch_size
    
    print(f"\nAfter weighting:")
    print(f"  cls_loss * hyp.cls * batch_size: {cls_loss_weighted.item():.2f}")
    
    print("\n" + "="*70)
    print("Analysis:")
    print("="*70)
    
    # The issue: summing over 8400 anchors
    avg_bce_per_anchor = bce_loss.sum() / (batch_size * num_anchors * nc)
    print(f"\nAverage BCE per anchor: {avg_bce_per_anchor.item():.4f}")
    print(f"Number of anchors: {num_anchors}")
    print(f"Total sum: {num_anchors} × {avg_bce_per_anchor.item():.4f} = {(num_anchors * avg_bce_per_anchor).item():.1f}")
    
    # With nc=1 and small target_scores_sum
    print(f"\nWith small target_scores_sum (~{target_scores_sum.item():.0f}):")
    print(f"  Loss = {bce_loss.sum().item():.0f} / {target_scores_sum.item():.0f} = {cls_loss.item():.1f}")
    
    if cls_loss.item() > 50:
        print(f"\n⚠️  Loss is HIGH because:")
        print(f"     - Summing over {num_anchors} anchors creates large numerator")
        print(f"     - Few positive samples creates small denominator")
        print(f"     - Result: {bce_loss.sum().item():.0f} / {target_scores_sum.item():.0f} = LARGE")
    
    print("\n" + "="*70)
    print("Testing with your actual training numbers:")
    print("="*70)
    
    # Simulate scenario with even fewer positives (your case)
    target_scores_sparse = torch.zeros(batch_size, num_anchors, nc)
    # Only 1-2 objects per image in your dataset
    instances_per_image_sparse = 2
    for b in range(batch_size):
        matched_indices = torch.randperm(num_anchors)[:instances_per_image_sparse]
        target_scores_sparse[b, matched_indices, 0] = torch.rand(instances_per_image_sparse) * 0.5 + 0.5
    
    bce_loss_sparse = bce(pred_scores, target_scores_sparse)
    target_scores_sum_sparse = max(target_scores_sparse.sum(), 1)
    cls_loss_sparse = bce_loss_sparse.sum() / target_scores_sum_sparse
    cls_loss_weighted_sparse = cls_loss_sparse * hyp_cls * batch_size
    
    print(f"\nWith fewer objects (~{instances_per_image_sparse} per image):")
    print(f"  target_scores_sum: {target_scores_sum_sparse.item():.2f}")
    print(f"  cls_loss (raw): {cls_loss_sparse.item():.2f}")
    print(f"  cls_loss (weighted): {cls_loss_weighted_sparse.item():.2f}")
    
    if cls_loss_weighted_sparse.item() > 500:
        print(f"\n🔴 THIS MATCHES YOUR TRAINING OUTPUT (~{cls_loss_weighted_sparse.item():.0f})!")
        print(f"\n  Root cause: {num_anchors} background anchors contribute to loss")
        print(f"              but only ~{instances_per_image_sparse*batch_size} positive anchors normalize it")
        print(f"              = Imbalance of {num_anchors/(instances_per_image_sparse*batch_size):.0f}:1")
    
    print("="*70)


if __name__ == '__main__':
    simulate_yolo_cls_loss()
