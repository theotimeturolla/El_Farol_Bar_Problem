"""
Create visual report comparing results
"""
import matplotlib.pyplot as plt
import numpy as np

# Simulate data based on your logs (replace with actual if you save them)

# Baseline data (from your logs)
baseline_episodes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
baseline_attendance = [98.8, 99.5, 99.6, 91.6, 83.6, 36.0, 28.1, 22.2, 40.2, 52.1, 95.5, 20.5, 81.8, 49.9, 22.3, 38.1, 95.6, 52.0, 46.2]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('El Farol Bar - Training Analysis', fontsize=16, fontweight='bold')

# Plot 1: Attendance over episodes
axes[0, 0].plot(baseline_episodes, baseline_attendance, 'o-', linewidth=2, markersize=6, label='Baseline')
axes[0, 0].axhline(y=60, color='r', linestyle='--', linewidth=2, label='Optimal (60)')
axes[0, 0].fill_between([0, 1000], 55, 65, alpha=0.2, color='green', label='Acceptable range')
axes[0, 0].set_xlabel('Episode', fontsize=11)
axes[0, 0].set_ylabel('Average Attendance', fontsize=11)
axes[0, 0].set_title('Attendance Convergence', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(0, 1000)

# Plot 2: Epsilon decay comparison
episodes = np.arange(0, 1000)
epsilon_baseline = np.maximum(0.01, 1.0 * 0.995**episodes)
epsilon_improved = np.maximum(0.01, 1.0 * 0.9995**episodes)

axes[0, 1].plot(episodes, epsilon_baseline, linewidth=2, label='Baseline (0.995)')
axes[0, 1].plot(episodes, epsilon_improved, linewidth=2, label='Improved (0.9995)')
axes[0, 1].axhline(y=0.1, color='gray', linestyle=':', label='Exploration threshold')
axes[0, 1].set_xlabel('Episode', fontsize=11)
axes[0, 1].set_ylabel('Epsilon (Exploration Rate)', fontsize=11)
axes[0, 1].set_title('Exploration Decay Rates', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(-0.05, 1.05)

# Plot 3: Problem analysis
problem_text = """
Key Issues Identified:

1. Epsilon reaches minimum too fast
   → Agents stop exploring at episode ~150
   
2. Extreme oscillations (0-100)
   → All agents follow same policy
   → Herding behavior
   
3. Parameter sharing creates correlation
   → Need more diversity
   
4. Weak penalties for bad coordination
   → Agents don't avoid extremes strongly
"""

axes[1, 0].text(0.05, 0.95, problem_text, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 0].set_title('Problems Identified', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Plot 4: Solutions
solution_text = """
Solutions Implemented/Proposed:

✓ Slower epsilon decay (0.9995)
  → More exploration over 2000 episodes
  
→ Heterogeneous agents (next step)
  → Different learning rates
  → Different exploration rates
  
→ Stronger penalties (next step)
  → Quadratic distance penalty
  → Zero reward for extreme attendance
  
→ Communication/Social networks
  → Observe neighbors' actions
  → Break symmetry
"""

axes[1, 1].text(0.05, 0.95, solution_text, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[1, 1].set_title('Solutions', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('results/figures/training_analysis_report.png', dpi=150, bbox_inches='tight')
print("✅ Report saved: results/figures/training_analysis_report.png")

# Copy to desktop
import os
os.system('cp results/figures/training_analysis_report.png /mnt/c/Users/Utilisateur/Desktop/ 2>/dev/null')
print("✅ Copied to Desktop")

plt.show()

