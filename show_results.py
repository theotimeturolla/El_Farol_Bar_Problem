"""
Quick script to display existing training results
"""
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 70)
print("ANALYZING EXISTING RESULTS")
print("=" * 70)

# Check what results exist
baseline_exists = os.path.exists('results/figures/baseline_training.png')
improved_exists = os.path.exists('results/improved/figures/baseline_training.png')

print(f"\n✓ Baseline results exist: {baseline_exists}")
print(f"✓ Improved results exist: {improved_exists}")

if not baseline_exists:
    print("\n❌ No baseline results found. Please run training first.")
    exit()

# From your training logs, recreate the data
print("\n" + "=" * 70)
print("BASELINE RESULTS (from your logs)")
print("=" * 70)

baseline_data = {
    50: {'reward': 61256.43, 'attendance': 98.8, 'epsilon': 0.010},
    100: {'reward': 60450.98, 'attendance': 99.5, 'epsilon': 0.010},
    150: {'reward': 60406.58, 'attendance': 99.6, 'epsilon': 0.010},
    200: {'reward': 59555.91, 'attendance': 91.6, 'epsilon': 0.010},
    250: {'reward': 58797.10, 'attendance': 83.6, 'epsilon': 0.010},
    300: {'reward': 53794.51, 'attendance': 36.0, 'epsilon': 0.010},
    350: {'reward': 53063.63, 'attendance': 28.1, 'epsilon': 0.010},
    400: {'reward': 52192.54, 'attendance': 22.2, 'epsilon': 0.010},
    450: {'reward': 53402.87, 'attendance': 40.2, 'epsilon': 0.010},
    500: {'reward': 54966.78, 'attendance': 52.1, 'epsilon': 0.010},
    550: {'reward': 60077.43, 'attendance': 95.5, 'epsilon': 0.010},
    600: {'reward': 51499.99, 'attendance': 20.5, 'epsilon': 0.010},
    650: {'reward': 59136.07, 'attendance': 81.8, 'epsilon': 0.010},
    700: {'reward': 55090.11, 'attendance': 49.9, 'epsilon': 0.010},
    750: {'reward': 52805.96, 'attendance': 22.3, 'epsilon': 0.010},
    800: {'reward': 53577.22, 'attendance': 38.1, 'epsilon': 0.010},
    850: {'reward': 60188.32, 'attendance': 95.6, 'epsilon': 0.010},
    900: {'reward': 55458.66, 'attendance': 52.0, 'epsilon': 0.010},
    950: {'reward': 54805.58, 'attendance': 46.2, 'epsilon': 0.010},
}

# Display results
print("\nEpisode | Avg Reward | Avg Attendance | Epsilon")
print("-" * 60)
for episode in sorted(baseline_data.keys()):
    data = baseline_data[episode]
    print(f"{episode:7d} | {data['reward']:10.2f} | {data['attendance']:14.1f} | {data['epsilon']:.3f}")

# Calculate statistics
attendances = [v['attendance'] for v in baseline_data.values()]

print("\n" + "=" * 70)
print("STATISTICS")
print("=" * 70)
print(f"Mean attendance: {np.mean(attendances):.2f}")
print(f"Std attendance:  {np.std(attendances):.2f}")
print(f"Min attendance:  {np.min(attendances):.1f}")
print(f"Max attendance:  {np.max(attendances):.1f}")

# Distance from optimal
distances = np.abs(np.array(attendances) - 60)
print(f"\nMean distance from optimal: {np.mean(distances):.2f}")

# % near optimal
within_5 = np.sum(distances <= 5)
within_10 = np.sum(distances <= 10)
print(f"% within ±5 of optimal:  {100*within_5/len(distances):.1f}%")
print(f"% within ±10 of optimal: {100*within_10/len(distances):.1f}%")

# Key observations
print("\n" + "=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)
print("1. Epsilon reached minimum (0.01) very quickly (~episode 150)")
print("2. Extreme oscillations: attendance swings from ~20 to ~100")
print("3. Never stabilizes around optimal (60)")
print("4. Anti-coordination dynamics: all go → none go → all go")
print("5. Parameter sharing creates herding behavior")

print("\n✅ Analysis complete!\n")
