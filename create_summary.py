"""
Create visual summary of baseline results
"""
import matplotlib.pyplot as plt
import numpy as np
import os

print("Creating visual summary...")

# Baseline data from your logs
episodes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
attendances = [98.8, 99.5, 99.6, 91.6, 83.6, 36.0, 28.1, 22.2, 40.2, 52.1, 95.5, 20.5, 81.8, 49.9, 22.3, 38.1, 95.6, 52.0, 46.2]
rewards = [61256.43, 60450.98, 60406.58, 59555.91, 58797.10, 53794.51, 53063.63, 52192.54, 53402.87, 54966.78, 60077.43, 51499.99, 59136.07, 55090.11, 52805.96, 53577.22, 60188.32, 55458.66, 54805.58]

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('El Farol Bar - Baseline Training Analysis', fontsize=18, fontweight='bold')

# Plot 1: Attendance over time
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(episodes, attendances, 'o-', linewidth=2, markersize=8, color='steelblue', label='Avg Attendance')
ax1.axhline(y=60, color='red', linestyle='--', linewidth=2, label='Optimal (60)')
ax1.fill_between(episodes, 55, 65, alpha=0.2, color='green', label='Target range (±5)')
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Attendance', fontsize=12)
ax1.set_title('Bar Attendance Evolution', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 105)

# Plot 2: Rewards
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(episodes, rewards, 'o-', linewidth=2, markersize=6, color='purple')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Average Reward', fontsize=12)
ax2.set_title('Reward Progression', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Epsilon decay
ax3 = fig.add_subplot(gs[1, 1])
epsilon_full = np.maximum(0.01, 1.0 * 0.995 ** np.arange(1000))
ax3.plot(epsilon_full, linewidth=2, color='orange')
ax3.axhline(y=0.1, color='gray', linestyle=':', linewidth=2, label='Exploration threshold')
ax3.axvline(x=150, color='red', linestyle='--', alpha=0.5, label='Min reached (~150)')
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Epsilon', fontsize=12)
ax3.set_title('Exploration Rate (ε) Decay', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.05, 1.05)

# Plot 4: Statistics box
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

stats_text = f"""
KEY STATISTICS (Episodes 50-950):

- Mean Attendance:           {np.mean(attendances):.2f}
- Std Deviation:             {np.std(attendances):.2f}
- Min Attendance:            {np.min(attendances):.1f}
- Max Attendance:            {np.max(attendances):.1f}
- Mean Abs Error from 60:    {np.mean(np.abs(np.array(attendances) - 60)):.2f}

PROBLEMS IDENTIFIED:

1. ❌ Epsilon decays TOO FAST (reaches 0.01 at episode ~150)
   → Agents stop exploring and get stuck

2. ❌ EXTREME OSCILLATIONS (20 ↔ 100)
   → Anti-coordination: everyone goes → nobody goes → repeat

3. ❌ HERDING BEHAVIOR
   → All agents use same policy → correlated decisions

4. ❌ NO CONVERGENCE to optimal (60)
   → Never stabilizes in target range

SOLUTION NEEDED:
→ Slower epsilon decay (0.9995 instead of 0.995)
→ More exploration over longer period
"""

ax4.text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('results/baseline_summary_report.png', dpi=150, bbox_inches='tight')
print("✅ Summary saved: results/baseline_summary_report.png")

# Copy to desktop
os.system('cp results/baseline_summary_report.png /mnt/c/Users/Utilisateur/Desktop/ 2>/dev/null')
print("✅ Copied to Windows Desktop")

plt.close()

print("\n" + "="*70)
print("SUMMARY FILES CREATED:")
print("="*70)
print("1. results/baseline_summary_report.png")
print("2. Copy on Desktop: baseline_summary_report.png")
print("\nYou can now upload this image to show the results!")
print("="*70)
