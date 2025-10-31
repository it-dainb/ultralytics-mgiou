"""
Real-time Training Monitor for Hybrid Loss
===========================================

Monitors a YOLO polygon training session and displays:
- Current alpha value (L2 vs MGIoU weight)
- Loss components over time
- Gradient flow statistics
- Visual plots updated in real-time

Usage:
    # Start training in one terminal
    yolo train model=yolo11n-polygon.yaml data=coco128.yaml use_hybrid=True
    
    # Monitor in another terminal
    python monitor_hybrid_training.py --run runs/polygon/train
    
    # Or specify log file directly
    python monitor_hybrid_training.py --log runs/polygon/train/results.csv
"""

import argparse
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class HybridLossMonitor:
    """Real-time monitor for hybrid loss training."""
    
    def __init__(self, log_path, refresh_rate=5.0):
        """
        Initialize monitor.
        
        Args:
            log_path: Path to training results.csv or training directory
            refresh_rate: Update interval in seconds
        """
        self.log_path = Path(log_path)
        if self.log_path.is_dir():
            self.log_path = self.log_path / 'results.csv'
        
        self.refresh_rate = refresh_rate
        self.last_epoch = -1
        
        # Initialize plot
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Hybrid Loss Training Monitor', fontsize=16, fontweight='bold')
        
        self.lines = {}
        self.data = {
            'epoch': [],
            'box_loss': [],
            'polygon_loss': [],
            'cls_loss': [],
            'dfl_loss': [],
            'map50': [],
            'map50_95': []
        }
    
    def read_log(self):
        """Read training log file."""
        if not self.log_path.exists():
            return None
        
        try:
            df = pd.read_csv(self.log_path)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            print(f"Error reading log: {e}")
            return None
    
    def calculate_alpha(self, epoch, total_epochs=100, schedule='cosine', 
                       alpha_start=0.9, alpha_end=0.2):
        """Calculate current alpha value based on schedule."""
        if epoch >= total_epochs:
            return alpha_end
        
        progress = epoch / (total_epochs - 1)
        
        if schedule == 'cosine':
            alpha = alpha_end + 0.5 * (alpha_start - alpha_end) * (1 + np.cos(np.pi * progress))
        elif schedule == 'linear':
            alpha = alpha_start + (alpha_end - alpha_start) * progress
        elif schedule == 'step':
            if progress < 0.5:
                alpha = alpha_start
            else:
                alpha = alpha_end
        else:
            alpha = alpha_start
        
        return alpha
    
    def update_plot(self, frame):
        """Update plots with latest data."""
        df = self.read_log()
        
        if df is None or len(df) == 0:
            return
        
        # Check for new epochs
        current_epoch = len(df) - 1
        if current_epoch <= self.last_epoch:
            return
        
        self.last_epoch = current_epoch
        
        # Update data
        self.data['epoch'] = df['epoch'].tolist()
        
        # Handle different possible column names
        for col_key, possible_names in [
            ('box_loss', ['train/box_loss', 'box_loss']),
            ('polygon_loss', ['train/polygon_loss', 'polygon_loss', 'train/kpt_loss']),
            ('cls_loss', ['train/cls_loss', 'cls_loss']),
            ('dfl_loss', ['train/dfl_loss', 'dfl_loss']),
            ('map50', ['metrics/mAP50(B)', 'val/map50', 'map50']),
            ('map50_95', ['metrics/mAP50-95(B)', 'val/map50_95', 'map50_95'])
        ]:
            for name in possible_names:
                if name in df.columns:
                    self.data[col_key] = df[name].tolist()
                    break
            else:
                self.data[col_key] = [0] * len(df)
        
        # Calculate alpha values
        alphas = [self.calculate_alpha(e) for e in self.data['epoch']]
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Alpha schedule
        ax = self.axes[0, 0]
        ax.plot(self.data['epoch'], alphas, 'b-', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Alpha (L2 Weight)', fontsize=11)
        ax.set_title('Alpha Schedule (L2→MGIoU Transition)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add current value annotation
        if len(alphas) > 0:
            current_alpha = alphas[-1]
            ax.annotate(f'Current: {current_alpha:.3f}',
                       xy=(self.data['epoch'][-1], current_alpha),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       fontweight='bold')
        
        # Plot 2: Loss components
        ax = self.axes[0, 1]
        if max(self.data['polygon_loss']) > 0:
            ax.plot(self.data['epoch'], self.data['polygon_loss'], 'r-', 
                   label='Polygon Loss', linewidth=2)
        if max(self.data['box_loss']) > 0:
            ax.plot(self.data['epoch'], self.data['box_loss'], 'b-', 
                   label='Box Loss', linewidth=2)
        if max(self.data['cls_loss']) > 0:
            ax.plot(self.data['epoch'], self.data['cls_loss'], 'g-', 
                   label='Class Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss Value', fontsize=11)
        ax.set_title('Loss Components', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        # Plot 3: mAP metrics
        ax = self.axes[1, 0]
        if max(self.data['map50_95']) > 0:
            ax.plot(self.data['epoch'], self.data['map50_95'], 'purple', 
                   label='mAP@50-95', linewidth=2, marker='o', markersize=3)
        if max(self.data['map50']) > 0:
            ax.plot(self.data['epoch'], self.data['map50'], 'orange', 
                   label='mAP@50', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('mAP', fontsize=11)
        ax.set_title('Validation Metrics', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        # Plot 4: Effective loss contributions
        ax = self.axes[1, 1]
        if len(alphas) > 0 and max(self.data['polygon_loss']) > 0:
            l2_contribution = [a * p for a, p in zip(alphas, self.data['polygon_loss'])]
            mgiou_contribution = [(1-a) * p for a, p in zip(alphas, self.data['polygon_loss'])]
            
            ax.plot(self.data['epoch'], l2_contribution, 'b-', 
                   label='L2 Component', linewidth=2)
            ax.plot(self.data['epoch'], mgiou_contribution, 'r-', 
                   label='MGIoU Component', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss Contribution', fontsize=11)
            ax.set_title('Hybrid Loss Breakdown', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
        
        self.fig.tight_layout()
    
    def run_console_mode(self):
        """Run in console mode without GUI."""
        print(f"\n{'='*80}")
        print("Hybrid Loss Training Monitor (Console Mode)")
        print(f"{'='*80}")
        print(f"Log file: {self.log_path}")
        print(f"Refresh rate: {self.refresh_rate}s")
        print(f"{'='*80}\n")
        
        try:
            while True:
                df = self.read_log()
                
                if df is None or len(df) == 0:
                    print(f"Waiting for training data... ({time.strftime('%H:%M:%S')})")
                    time.sleep(self.refresh_rate)
                    continue
                
                current_epoch = len(df) - 1
                
                if current_epoch > self.last_epoch:
                    self.last_epoch = current_epoch
                    
                    # Calculate alpha
                    alpha = self.calculate_alpha(current_epoch)
                    
                    # Extract latest metrics
                    latest = df.iloc[-1]
                    
                    # Format output
                    print(f"\n{'='*80}")
                    print(f"Epoch {int(latest.get('epoch', current_epoch))}")
                    print(f"{'='*80}")
                    print(f"  Alpha:          {alpha:.4f} (L2: {alpha:.1%}, MGIoU: {(1-alpha):.1%})")
                    
                    # Loss metrics
                    poly_loss = latest.get('train/polygon_loss', latest.get('polygon_loss', 0))
                    box_loss = latest.get('train/box_loss', latest.get('box_loss', 0))
                    cls_loss = latest.get('train/cls_loss', latest.get('cls_loss', 0))
                    
                    print(f"\n  Losses:")
                    print(f"    Polygon:      {poly_loss:.6f}")
                    print(f"    Box:          {box_loss:.6f}")
                    print(f"    Class:        {cls_loss:.6f}")
                    
                    # Validation metrics
                    map50_95 = latest.get('metrics/mAP50-95(B)', latest.get('val/map50_95', 0))
                    map50 = latest.get('metrics/mAP50(B)', latest.get('val/map50', 0))
                    
                    if map50_95 > 0 or map50 > 0:
                        print(f"\n  Validation:")
                        print(f"    mAP@50-95:    {map50_95:.4f}")
                        print(f"    mAP@50:       {map50:.4f}")
                    
                    # Effective contributions
                    l2_contrib = alpha * poly_loss
                    mgiou_contrib = (1 - alpha) * poly_loss
                    
                    print(f"\n  Polygon Loss Breakdown:")
                    print(f"    L2 contrib:   {l2_contrib:.6f} ({alpha:.1%})")
                    print(f"    MGIoU contrib:{mgiou_contrib:.6f} ({(1-alpha):.1%})")
                
                time.sleep(self.refresh_rate)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
    
    def run_gui_mode(self):
        """Run in GUI mode with live plots."""
        print(f"\n{'='*80}")
        print("Hybrid Loss Training Monitor (GUI Mode)")
        print(f"{'='*80}")
        print(f"Log file: {self.log_path}")
        print(f"Refresh rate: {self.refresh_rate}s")
        print(f"{'='*80}\n")
        print("Close the plot window to stop monitoring.\n")
        
        # Set up animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=int(self.refresh_rate * 1000))
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Monitor hybrid loss training in real-time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor by training directory
  python monitor_hybrid_training.py --run runs/polygon/train
  
  # Monitor by log file
  python monitor_hybrid_training.py --log runs/polygon/train/results.csv
  
  # Console mode (no GUI)
  python monitor_hybrid_training.py --run runs/polygon/train --console
  
  # Custom refresh rate
  python monitor_hybrid_training.py --run runs/polygon/train --refresh 2
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', type=str, 
                      help='Training run directory (contains results.csv)')
    group.add_argument('--log', type=str,
                      help='Direct path to results.csv file')
    
    parser.add_argument('--refresh', type=float, default=5.0,
                       help='Refresh rate in seconds (default: 5.0)')
    parser.add_argument('--console', action='store_true',
                       help='Console mode (no GUI)')
    parser.add_argument('--total-epochs', type=int, default=100,
                       help='Total training epochs (for alpha calculation)')
    parser.add_argument('--schedule', type=str, default='cosine',
                       choices=['cosine', 'linear', 'step'],
                       help='Alpha schedule type (default: cosine)')
    parser.add_argument('--alpha-start', type=float, default=0.9,
                       help='Starting alpha value (default: 0.9)')
    parser.add_argument('--alpha-end', type=float, default=0.2,
                       help='Ending alpha value (default: 0.2)')
    
    args = parser.parse_args()
    
    # Determine log path
    log_path = args.log if args.log else args.run
    
    # Check if path exists
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Error: Path does not exist: {log_path}")
        print("\nWaiting for training to start...")
        
        # Wait for training to create the file
        parent = log_path if log_path.is_dir() else log_path.parent
        while not parent.exists():
            time.sleep(1)
    
    # Create monitor
    monitor = HybridLossMonitor(log_path, refresh_rate=args.refresh)
    
    # Override alpha calculation parameters if provided
    original_calc = monitor.calculate_alpha
    def wrapped_calc(epoch, **kwargs):
        kwargs.setdefault('total_epochs', args.total_epochs)
        kwargs.setdefault('schedule', args.schedule)
        kwargs.setdefault('alpha_start', args.alpha_start)
        kwargs.setdefault('alpha_end', args.alpha_end)
        return original_calc(epoch, **kwargs)
    monitor.calculate_alpha = wrapped_calc
    
    # Run in appropriate mode
    if args.console:
        monitor.run_console_mode()
    else:
        try:
            monitor.run_gui_mode()
        except Exception as e:
            print(f"\nGUI mode failed: {e}")
            print("Falling back to console mode...\n")
            monitor.run_console_mode()


if __name__ == '__main__':
    main()
