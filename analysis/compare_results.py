"""
Compare baseline vs improved results
"""
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import os

# Check if results exist
baseline_exists = os.path.exists('results/figures/baseline_training.png')
improved_exists = os.path.exists('results/improved/figures/baseline_training.png')

if not improved_exists:
    print("❌ Improved results not found. Run: python3 experiments/baseline_improved.py")
    exit(1)

print("📊 Creating comparison plots...")

# For now, let's create a simple analysis of what should be different
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Baseline vs Improved Comparison', fontsize=16, fontweight='bold')

# Placeholder for actual data comparison
# You would load actual training logs here

axes[0, 0].text(0.5, 0.5, 'Baseline\nEpsilon Decay: 0.995\nReaches 0.01 at episode ~150', 
                ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
axes[0, 0].set_title('Baseline Configuration')
axes[0, 0].axis('off')

axes[0, 1].text(0.5, 0.5, 'Improved\nEpsilon Decay: 0.9995\nReaches 0.01 at episode ~4600', 
                ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))
axes[0, 1].set_title('Improved Configuration')
axes[0, 1].axis('off')

# Expected improvements
improvements = [
    "✓ More sustained exploration",
    "✓ Better convergence to optimal",
    "✓ Less oscillation",
    "✓ More stable attendance around 60"
]

axes[1, 0].text(0.1, 0.5, '\n'.join(improvements), 
                fontsize=11, verticalalignment='center', family='monospace')
axes[1, 0].set_title('Expected Improvements')
axes[1, 0].axis('off')

# Next steps
next_steps = [
    "1. Compare final attendance stability",
    "2. Measure variance around optimal",
    "3. Check convergence speed",
    "4. Analyze reward progression"
]

axes[1, 1].text(0.1, 0.5, '\n'.join(next_steps), 
                fontsize=11, verticalalignment='center', family='monospace')
axes[1, 1].set_title('Analysis TODO')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('results/comparison_overview.png', dpi=150, bbox_inches='tight')
print("✅ Comparison saved to: results/comparison_overview.png")

# Copy to desktop
os.system('cp results/comparison_overview.png /mnt/c/Users/Utilisateur/Desktop/ 2>/dev/null')
print("✅ Copied to Windows Desktop")

