"""
Analyse Multi-Bars Results
"""
import matplotlib.pyplot as plt
import numpy as np
import os

print("="*80)
print("MULTI-BARS ANALYSIS")
print("="*80)

# From training logs
episodes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
bar0_att = [8.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5]
bar1_att = [8.7, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5]
bar2_att = [8.4, 0.5, 0.4, 0.5, 0.4, 0.6, 0.4, 0.4, 0.4, 0.4]

bar0_price = [14.77, 11.21, 7.63, 14.61, 14.13, 14.09, 13.54, 13.96, 14.17, 14.77]
bar1_price = [14.77, 14.72, 14.80, 14.78, 14.78, 14.80, 14.78, 14.68, 14.78, 14.78]
bar2_price = [14.79, 14.75, 14.77, 14.77, 14.75, 14.75, 14.73, 14.78, 14.79, 14.79]

consumer_reward = [36739, 87178, 87402, 86899, 86963, 86978, 86978, 86915, 86952, 86978]

print("\nKEY FINDINGS:")
print("-"*80)
print(f"Average Attendance per Bar: {np.mean(bar0_att + bar1_att + bar2_att):.2f}")
print(f"Total Attendance: {np.mean(bar0_att + bar1_att + bar2_att) * 3:.2f} / 180")
print(f"Average Price: ${np.mean(bar0_price + bar1_price + bar2_price):.2f}")
print(f"Consumer Reward (avg): {np.mean(consumer_reward):.0f}")
print()
print("⚠️  PROBLEM IDENTIFIED:")
print("  - Attendance collapsed to ~0.5 per bar (~1.5 total out of 180!)")
print("  - 178.5 consumers stay home on average")
print("  - High consumer rewards = staying home is optimal")
print("  - Same 'Conservative Learning Trap' as Improved baseline")
print("="*80)

# Create summary
with open('results/multi_bars_180/analysis.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("MULTI-BARS TRAINING ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write("CONFIGURATION:\n")
    f.write("  Episodes: 1000\n")
    f.write("  Consumers: 180\n")
    f.write("  Bars: 3 (optimal capacity: 60 each)\n")
    f.write("  Price sensitivity: ×1.5\n\n")
    
    f.write("RESULTS:\n")
    f.write(f"  Avg Attendance/Bar: {np.mean(bar0_att + bar1_att + bar2_att):.2f}\n")
    f.write(f"  Total Attendance: {np.mean(bar0_att + bar1_att + bar2_att) * 3:.2f} / 180\n")
    f.write(f"  Avg Price: ${np.mean(bar0_price + bar1_price + bar2_price):.2f}\n")
    f.write(f"  Consumer Reward: {np.mean(consumer_reward):.0f}\n\n")
    
    f.write("PROBLEM:\n")
    f.write("  Conservative Learning Trap (same as Improved baseline)\n")
    f.write("  - Consumers learned staying home = optimal\n")
    f.write("  - Bars have almost no customers (<1 person)\n")
    f.write("  - Prices remain at maximum (no competition pressure)\n\n")
    
    f.write("ROOT CAUSE:\n")
    f.write("  1. Consumer epsilon reached 0.01 too quickly\n")
    f.write("  2. Staying home reward (5.0) is safer than risky coordination\n")
    f.write("  3. Empty bars → bad experience → reinforces staying home\n")
    f.write("  4. Circular trap: nobody goes → nobody wants to go\n\n")
    
    f.write("IMPLICATIONS FOR REPORT:\n")
    f.write("  - Demonstrates coordination failure in multi-agent RL\n")
    f.write("  - Risk-averse convergence in complex games\n")
    f.write("  - Need for reward engineering or forced exploration\n")

print("✅ Analysis saved: results/multi_bars_180/analysis.txt")
print()

