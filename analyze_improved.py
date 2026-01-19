"""
Analyse complète : Baseline vs Improved
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*80)
print("ANALYSE : BASELINE vs IMPROVED")
print("="*80)

# Charger improved data
with open('results/improved/data/improved_data.pkl', 'rb') as f:
    improved = pickle.load(f)

# Baseline stats (de vos résultats précédents)
baseline_stats = {
    'mean_attendance': 60.72,
    'std_attendance': 29.45,
    'within_5': 0.0,
    'within_10': 10.5,
    'within_20': 26.3,
    'epsilon_final': 0.010,
    'mean_reward': 56343.51
}

# Analyser improved (derniers 200 épisodes)
improved_att = improved['episode_attendances'][-200:]
improved_rewards = improved['episode_rewards'][-200:]
improved_eps = improved['epsilon_history']

mean_att = np.mean(improved_att)
std_att = np.std(improved_att)
median_att = np.median(improved_att)
mean_reward = np.mean(improved_rewards)

distances = np.abs(np.array(improved_att) - 60)
mae = np.mean(distances)
within_5 = 100 * np.sum(distances <= 5) / len(distances)
within_10 = 100 * np.sum(distances <= 10) / len(distances)
within_20 = 100 * np.sum(distances <= 20) / len(distances)

improved_stats = {
    'mean_attendance': mean_att,
    'std_attendance': std_att,
    'median_attendance': median_att,
    'within_5': within_5,
    'within_10': within_10,
    'within_20': within_20,
    'epsilon_final': improved_eps[-1],
    'mean_reward': mean_reward,
    'mae': mae
}

# Afficher comparaison
print("\n" + "="*80)
print("COMPARAISON QUANTITATIVE (Derniers 200 épisodes)")
print("="*80)
print(f"{'Métrique':<35} {'Baseline':<20} {'Improved':<20} {'Δ':<15}")
print("-"*80)

metrics = [
    ('Mean Attendance', 'mean_attendance', '.2f'),
    ('Std Deviation', 'std_attendance', '.2f'),
    ('Median Attendance', 'median_attendance', '.2f'),
    ('% within ±5', 'within_5', '.1f'),
    ('% within ±10', 'within_10', '.1f'),
    ('% within ±20', 'within_20', '.1f'),
    ('Mean Reward', 'mean_reward', '.2f'),
    ('Final Epsilon', 'epsilon_final', '.4f'),
]

for label, key, fmt in metrics:
    if key == 'median_attendance':
        baseline_val = 52.0  # De vos logs
        improved_val = improved_stats[key]
        delta = improved_val - baseline_val
    else:
        baseline_val = baseline_stats[key]
        improved_val = improved_stats[key]
        delta = improved_val - baseline_val
    
    baseline_str = f"{baseline_val:{fmt}}"
    improved_str = f"{improved_val:{fmt}}"
    delta_str = f"{delta:+{fmt}}"
    
    print(f"{label:<35} {baseline_str:<20} {improved_str:<20} {delta_str:<15}")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

# Calcul du score d'amélioration
score = 0

if std_att < baseline_stats['std_attendance'] * 0.6:
    print("✅ EXCELLENTE réduction de la variance (>40%)")
    score += 3
elif std_att < baseline_stats['std_attendance'] * 0.8:
    print("🟢 Bonne réduction de la variance (20-40%)")
    score += 2
elif std_att < baseline_stats['std_attendance']:
    print("🟡 Légère réduction de la variance (<20%)")
    score += 1
else:
    print("❌ Pas de réduction de variance")

if within_5 > 40:
    print("✅ EXCELLENTE coordination (>40% dans ±5)")
    score += 3
elif within_5 > 25:
    print("🟢 Bonne coordination (25-40% dans ±5)")
    score += 2
elif within_5 > 10:
    print("🟡 Coordination modérée (10-25% dans ±5)")
    score += 1
else:
    print("❌ Coordination insuffisante (<10% dans ±5)")

if mae < 10:
    print("✅ EXCELLENTE précision (MAE < 10)")
    score += 2
elif mae < 15:
    print("🟢 Bonne précision (MAE < 15)")
    score += 1

print()
if score >= 7:
    print("🎉 SUCCÈS MAJEUR : Epsilon lent améliore dramatiquement la coordination!")
elif score >= 5:
    print("✅ SUCCÈS : Amélioration significative par rapport au baseline")
elif score >= 3:
    print("🟡 SUCCÈS PARTIEL : Amélioration mais peut encore mieux faire")
else:
    print("❌ ÉCHEC : Pas d'amélioration notable")

print("="*80)

# Sauvegarder le résumé
with open('results/improved/summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("IMPROVED BASELINE - SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Episodes: 2000\n")
    f.write(f"  Epsilon decay: 0.9995\n")
    f.write(f"  Final epsilon: {improved_eps[-1]:.4f}\n\n")
    
    f.write("RESULTS (Last 200 episodes):\n")
    for label, key, fmt in metrics:
        if key in improved_stats:
            f.write(f"  {label}: {improved_stats[key]:{fmt}}\n")
    
    f.write(f"\nVS BASELINE:\n")
    f.write(f"  Std reduction: {((baseline_stats['std_attendance'] - std_att) / baseline_stats['std_attendance'] * 100):.1f}%\n")
    f.write(f"  Coordination improvement: {within_5 - baseline_stats['within_5']:.1f} percentage points\n")

print("\n✅ Résumé sauvegardé : results/improved/summary.txt")

# Créer graphique de comparaison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Baseline vs Improved Comparison', fontsize=16, fontweight='bold')

# Plot 1: Distribution attendance
baseline_att_sample = [98.8, 99.5, 99.6, 91.6, 83.6, 36.0, 28.1, 22.2, 40.2, 52.1, 95.5, 20.5, 81.8, 49.9, 22.3, 38.1, 95.6, 52.0, 46.2]

axes[0, 0].hist(baseline_att_sample, bins=15, alpha=0.6, label='Baseline', color='blue')
axes[0, 0].hist(improved_att, bins=15, alpha=0.6, label='Improved', color='green')
axes[0, 0].axvline(x=60, color='r', linestyle='--', linewidth=2, label='Optimal')
axes[0, 0].set_xlabel('Attendance')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Attendance Distribution')
axes[0, 0].legend()

# Plot 2: Epsilon comparison
eps_baseline = np.maximum(0.01, 1.0 * 0.995 ** np.arange(2000))
axes[0, 1].plot(eps_baseline, label='Baseline (0.995)', linewidth=2, color='blue')
axes[0, 1].plot(improved_eps, label='Improved (0.9995)', linewidth=2, color='green')
axes[0, 1].axhline(y=0.1, color='gray', linestyle=':', label='Threshold')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Epsilon')
axes[0, 1].set_title('Exploration Decay Rate')
axes[0, 1].legend()
axes[0, 1].set_ylim(-0.05, 1.05)

# Plot 3: Stats comparison
categories = ['Mean\nAtt', 'Std\nDev', '%\nWithin ±5', '%\nWithin ±10']
baseline_vals = [baseline_stats['mean_attendance'], baseline_stats['std_attendance'], 
                 baseline_stats['within_5'], baseline_stats['within_10']]
improved_vals = [improved_stats['mean_attendance'], improved_stats['std_attendance'],
                 improved_stats['within_5'], improved_stats['within_10']]

x = np.arange(len(categories))
width = 0.35

axes[1, 0].bar(x - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
axes[1, 0].bar(x + width/2, improved_vals, width, label='Improved', color='green', alpha=0.7)
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Key Metrics Comparison')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(categories)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Improvement summary
improvements = [
    f"Std: -{((baseline_stats['std_attendance'] - std_att) / baseline_stats['std_attendance'] * 100):.1f}%",
    f"±5: +{within_5 - baseline_stats['within_5']:.1f}pp",
    f"±10: +{within_10 - baseline_stats['within_10']:.1f}pp",
    f"Reward: +{mean_reward - baseline_stats['mean_reward']:.0f}"
]

axes[1, 1].text(0.1, 0.5, "KEY IMPROVEMENTS:\n\n" + "\n".join([f"• {imp}" for imp in improvements]),
               fontsize=14, verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
axes[1, 1].axis('off')
axes[1, 1].set_title('Summary')

plt.tight_layout()
plt.savefig('results/comparison_baseline_vs_improved.png', dpi=150, bbox_inches='tight')
print("✅ Graphique comparatif : results/comparison_baseline_vs_improved.png")

os.system('cp results/comparison_baseline_vs_improved.png /mnt/c/Users/Utilisateur/Desktop/')
print("✅ Copié sur Desktop!")

plt.close()

print("\n" + "="*80)
print("FICHIERS GÉNÉRÉS:")
print("="*80)
print("1. results/improved/summary.txt")
print("2. results/comparison_baseline_vs_improved.png (+ sur Desktop)")
print("3. results/improved/figures/improved_training.png (déjà généré)")
print("="*80)

