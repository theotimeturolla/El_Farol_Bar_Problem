"""
Training Multi-Bars with 180 consumers (3 bars × 60 optimal)
"""
import sys
sys.path.append('.')
from experiments.train_multi_bars import train_multi_bars

print("="*70)
print("MULTI-BARS TRAINING - 180 CONSUMERS")
print("="*70)
print("Configuration:")
print("  - 3 bars with capacity_optimal=60 each")
print("  - 180 consumers (60 per bar on average)")
print("  - Increased price sensitivity (×1.5)")
print("="*70)
print()

# Train with more consumers
train_multi_bars(
    n_episodes=1000,
    n_consumers=180,  # 3 bars × 60 optimal
    n_bars=3,
    batch_size=64,
    save_dir="results/multi_bars_180"
)
