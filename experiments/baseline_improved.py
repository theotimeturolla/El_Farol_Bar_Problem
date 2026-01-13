"""
Improved baseline with slower epsilon decay
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train_baseline

# Train with better hyperparameters
agent, rewards, attendances = train_baseline(
    n_episodes=2000,  # Plus d'épisodes
    n_consumers=100,
    optimal_attendance=60,
    batch_size=64,
    target_update_freq=10,
    save_dir="results/improved"
)
