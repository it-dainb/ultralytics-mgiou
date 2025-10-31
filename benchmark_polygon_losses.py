"""
Benchmark Script for YOLO Polygon Loss Strategies
==================================================

Compares three training strategies on the same dataset:
1. L2 Only: Traditional L2 loss for all epochs
2. Two-Stage: L2 for 70 epochs, then MGIoU for 50 epochs  
3. Hybrid: Dynamic blend of L2 and MGIoU with cosine schedule

Usage:
    python benchmark_polygon_losses.py --data coco128.yaml --epochs 100 --model yolo11n-polygon.yaml
    
Output:
    - CSV file with metrics comparison
    - Loss convergence plots
    - Training time comparison
    - Best weights for each strategy
"""

import argparse
import csv
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

def run_strategy(strategy_name, model_path, data_path, epochs, project, **kwargs):
    """Run a single training strategy and return metrics."""
    print(f"\n{'='*80}")
    print(f"Running Strategy: {strategy_name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Initialize model
    model = YOLO(model_path)
    
    # Train with specific strategy parameters
    results = model.train(
        data=data_path,
        epochs=epochs,
        project=project,
        name=strategy_name,
        **kwargs
    )
    
    training_time = time.time() - start_time
    
    # Extract metrics
    metrics = {
        'strategy': strategy_name,
        'training_time_seconds': training_time,
        'training_time_hours': training_time / 3600,
        'final_map50': results.results_dict.get('metrics/mAP50(B)', 0),
        'final_map50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
        'final_box_loss': results.results_dict.get('train/box_loss', 0),
        'final_polygon_loss': results.results_dict.get('train/polygon_loss', 0),
        'weights_path': str(Path(project) / strategy_name / 'weights' / 'best.pt')
    }
    
    print(f"\n✓ {strategy_name} completed in {training_time/3600:.2f} hours")
    print(f"  Final mAP50-95: {metrics['final_map50_95']:.4f}")
    print(f"  Final mAP50: {metrics['final_map50']:.4f}")
    
    return metrics, results


def plot_comparison(results_list, output_dir):
    """Generate comparison plots for all strategies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Training time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = [r['strategy'] for r in results_list]
    times = [r['training_time_hours'] for r in results_list]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    ax.bar(strategies, times, color=colors[:len(strategies)])
    ax.set_ylabel('Training Time (hours)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (s, t) in enumerate(zip(strategies, times)):
        ax.text(i, t + 0.1, f'{t:.2f}h', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_comparison.png', dpi=150)
    print(f"  ✓ Saved: {output_dir / 'training_time_comparison.png'}")
    plt.close()
    
    # Plot 2: mAP comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    map50 = [r['final_map50'] for r in results_list]
    map50_95 = [r['final_map50_95'] for r in results_list]
    
    ax1.bar(strategies, map50, color=colors[:len(strategies)])
    ax1.set_ylabel('mAP@50', fontsize=12)
    ax1.set_title('mAP@50 Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (s, m) in enumerate(zip(strategies, map50)):
        ax1.text(i, m + 0.01, f'{m:.4f}', ha='center', fontweight='bold')
    
    ax2.bar(strategies, map50_95, color=colors[:len(strategies)])
    ax2.set_ylabel('mAP@50-95', fontsize=12)
    ax2.set_title('mAP@50-95 Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (s, m) in enumerate(zip(strategies, map50_95)):
        ax2.text(i, m + 0.01, f'{m:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'map_comparison.png', dpi=150)
    print(f"  ✓ Saved: {output_dir / 'map_comparison.png'}")
    plt.close()
    
    # Plot 3: Loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    poly_losses = [r['final_polygon_loss'] for r in results_list]
    
    ax.bar(strategies, poly_losses, color=colors[:len(strategies)])
    ax.set_ylabel('Final Polygon Loss', fontsize=12)
    ax.set_title('Polygon Loss Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (s, l) in enumerate(zip(strategies, poly_losses)):
        ax.text(i, l + 0.001, f'{l:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=150)
    print(f"  ✓ Saved: {output_dir / 'loss_comparison.png'}")
    plt.close()


def save_results_csv(results_list, output_path):
    """Save benchmark results to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)
    
    print(f"\n✓ Results saved to: {output_path}")


def print_summary(results_list):
    """Print formatted summary of results."""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    # Find best strategy for each metric
    best_map50_95 = max(results_list, key=lambda x: x['final_map50_95'])
    best_time = min(results_list, key=lambda x: x['training_time_hours'])
    best_loss = min(results_list, key=lambda x: x['final_polygon_loss'])
    
    print("Best Performance Metrics:")
    print(f"  🏆 Highest mAP@50-95: {best_map50_95['strategy']} ({best_map50_95['final_map50_95']:.4f})")
    print(f"  ⚡ Fastest Training: {best_time['strategy']} ({best_time['training_time_hours']:.2f}h)")
    print(f"  📉 Lowest Loss: {best_loss['strategy']} ({best_loss['final_polygon_loss']:.4f})")
    
    print("\nDetailed Results:")
    print(f"{'Strategy':<15} {'mAP@50-95':<12} {'mAP@50':<12} {'Time(h)':<10} {'Poly Loss':<12}")
    print("-" * 65)
    for r in results_list:
        print(f"{r['strategy']:<15} {r['final_map50_95']:<12.4f} {r['final_map50']:<12.4f} "
              f"{r['training_time_hours']:<10.2f} {r['final_polygon_loss']:<12.4f}")
    
    print("\nWeights Locations:")
    for r in results_list:
        print(f"  {r['strategy']:<15} -> {r['weights_path']}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark YOLO Polygon Loss Strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with default settings
  python benchmark_polygon_losses.py --data coco128.yaml
  
  # Custom model and epochs
  python benchmark_polygon_losses.py --data my_data.yaml --epochs 150 --model yolo11s-polygon.yaml
  
  # Run specific strategies only
  python benchmark_polygon_losses.py --data coco128.yaml --strategies l2 hybrid
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='yolo11n-polygon.yaml',
                        help='Model configuration (default: yolo11n-polygon.yaml)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs (default: 100)')
    parser.add_argument('--project', type=str, default='runs/polygon_benchmark',
                        help='Project directory (default: runs/polygon_benchmark)')
    parser.add_argument('--strategies', nargs='+', 
                        choices=['l2', 'two_stage', 'hybrid'],
                        default=['l2', 'two_stage', 'hybrid'],
                        help='Strategies to benchmark (default: all)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (default: 0)')
    
    args = parser.parse_args()
    
    # Calculate epochs for two-stage strategy
    stage1_epochs = int(args.epochs * 0.7)
    stage2_epochs = args.epochs - stage1_epochs
    
    # Common training parameters
    common_params = {
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'verbose': True,
    }
    
    # Strategy configurations
    strategies = {
        'l2': {
            'name': 'L2_Only',
            'params': {
                'use_mgiou': False,
                'use_hybrid': False,
                **common_params
            }
        },
        'two_stage': {
            'name': 'Two_Stage',
            'stages': [
                {
                    'name': 'Two_Stage_L2',
                    'epochs': stage1_epochs,
                    'params': {
                        'use_mgiou': False,
                        'use_hybrid': False,
                        **common_params
                    }
                },
                {
                    'name': 'Two_Stage_MGIoU',
                    'epochs': stage2_epochs,
                    'params': {
                        'use_mgiou': True,
                        'use_hybrid': False,
                        **common_params
                    }
                }
            ]
        },
        'hybrid': {
            'name': 'Hybrid',
            'params': {
                'use_hybrid': True,
                'alpha_schedule': 'cosine',
                'alpha_start': 0.9,
                'alpha_end': 0.2,
                **common_params
            }
        }
    }
    
    results_list = []
    
    print(f"\n{'='*80}")
    print("POLYGON LOSS BENCHMARK")
    print(f"{'='*80}")
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Total Epochs: {args.epochs}")
    print(f"Strategies: {', '.join(args.strategies)}")
    print(f"{'='*80}\n")
    
    # Run each selected strategy
    for strategy_key in args.strategies:
        strategy = strategies[strategy_key]
        
        if strategy_key == 'two_stage':
            # Two-stage requires sequential training
            print(f"\n{'='*80}")
            print(f"Running Two-Stage Strategy")
            print(f"  Stage 1: L2 Loss for {stage1_epochs} epochs")
            print(f"  Stage 2: MGIoU Loss for {stage2_epochs} epochs")
            print(f"{'='*80}\n")
            
            start_time = time.time()
            
            # Stage 1: L2
            model = YOLO(args.model)
            results1 = model.train(
                data=args.data,
                epochs=stage1_epochs,
                project=args.project,
                name=strategy['stages'][0]['name'],
                **strategy['stages'][0]['params']
            )
            
            # Stage 2: Resume with MGIoU
            stage1_weights = Path(args.project) / strategy['stages'][0]['name'] / 'weights' / 'last.pt'
            model = YOLO(stage1_weights)
            results2 = model.train(
                data=args.data,
                epochs=stage2_epochs,
                project=args.project,
                name=strategy['stages'][1]['name'],
                **strategy['stages'][1]['params']
            )
            
            training_time = time.time() - start_time
            
            # Use final stage results
            metrics = {
                'strategy': 'Two_Stage',
                'training_time_seconds': training_time,
                'training_time_hours': training_time / 3600,
                'final_map50': results2.results_dict.get('metrics/mAP50(B)', 0),
                'final_map50_95': results2.results_dict.get('metrics/mAP50-95(B)', 0),
                'final_box_loss': results2.results_dict.get('train/box_loss', 0),
                'final_polygon_loss': results2.results_dict.get('train/polygon_loss', 0),
                'weights_path': str(Path(args.project) / strategy['stages'][1]['name'] / 'weights' / 'best.pt')
            }
            
            print(f"\n✓ Two-Stage completed in {training_time/3600:.2f} hours")
            print(f"  Final mAP50-95: {metrics['final_map50_95']:.4f}")
            
        else:
            # Single-stage strategies (L2 or Hybrid)
            metrics, _ = run_strategy(
                strategy['name'],
                args.model,
                args.data,
                args.epochs,
                args.project,
                **strategy['params']
            )
        
        results_list.append(metrics)
    
    # Generate outputs
    output_dir = Path(args.project) / 'benchmark_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Generating Comparison Plots...")
    print(f"{'='*80}\n")
    plot_comparison(results_list, output_dir)
    
    # Save CSV
    csv_path = output_dir / 'benchmark_results.csv'
    save_results_csv(results_list, csv_path)
    
    # Print summary
    print_summary(results_list)
    
    print(f"All outputs saved to: {output_dir}")
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
