"""
Improved Baseline Training with Slower Epsilon Decay
"""
import sys
sys.path.append('.')

from environment.el_farol_env import ElFarolEnv
from agents.dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

print("="*70)
print("IMPROVED BASELINE TRAINING")
print("="*70)
print("Key Change: Epsilon decay = 0.9995 (instead of 0.995)")
print("Episodes: 2000 (instead of 1000)")
print("Expected: Better coordination, less oscillation")
print("="*70)
print()

# Create environment
env = ElFarolEnv(n_consumers=100, optimal_attendance=60)

# Create agent with SLOWER epsilon decay
print("Creating DQN agent with epsilon_decay=0.9995...")
agent = DQNAgent(
    state_dim=5, 
    action_dim=2,
    lr=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.9995,  # ← KEY CHANGE: 0.9995 instead of 0.995
    buffer_size=10000
)

print("Agent created successfully!")
print()

# Tracking
episode_rewards = []
episode_attendances = []
epsilon_history = []

# Training loop
print("Starting training...")
for episode in tqdm(range(2000), desc="Training Progress"):
    observations, _ = env.reset()
    episode_reward = 0
    
    for step in range(100):
        # Select actions for all agents
        actions = {}
        for agent_id in env.agents:
            obs = observations[agent_id]
            action = agent.select_action(obs, training=True)
            actions[agent_id] = action
        
        # Step environment
        next_observations, rewards, dones, truncs, infos = env.step(actions)
        
        # Store transitions
        for agent_id in env.agents:
            agent.store_transition(
                observations[agent_id],
                actions[agent_id],
                rewards[agent_id],
                next_observations[agent_id],
                dones[agent_id]
            )
        
        # Update agent
        loss = agent.update(batch_size=64)
        episode_reward += sum(rewards.values())
        observations = next_observations
        
        if all(dones.values()):
            break
    
    # Update target network
    if episode % 10 == 0:
        agent.update_target_network()
    
    # Track metrics
    final_attendance = env.attendance_history[-1]
    episode_rewards.append(episode_reward)
    episode_attendances.append(final_attendance)
    epsilon_history.append(agent.epsilon)
    
    # Logging every 100 episodes
    if episode % 100 == 0 and episode > 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_attendance = np.mean(episode_attendances[-100:])
        std_attendance = np.std(episode_attendances[-100:])
        
        print(f"\n{'='*70}")
        print(f"Episode {episode}/2000")
        print(f"  Avg Reward (last 100):      {avg_reward:.2f}")
        print(f"  Avg Attendance (last 100):  {avg_attendance:.1f} ± {std_attendance:.1f}")
        print(f"  Epsilon:                    {agent.epsilon:.4f}")
        print(f"{'='*70}")

print()
print("="*70)
print("TRAINING COMPLETED!")
print("="*70)

# Save model
os.makedirs('results/improved/models', exist_ok=True)
os.makedirs('results/improved/figures', exist_ok=True)
os.makedirs('results/improved/data', exist_ok=True)

agent.save('results/improved/models/improved_dqn.pt')
print("✅ Model saved to: results/improved/models/improved_dqn.pt")

# Save data
import pickle
data = {
    'episode_rewards': episode_rewards,
    'episode_attendances': episode_attendances,
    'epsilon_history': epsilon_history,
    'config': {
        'epsilon_decay': 0.9995,
        'n_episodes': 2000,
        'n_consumers': 100,
        'optimal_attendance': 60
    }
}

with open('results/improved/data/improved_data.pkl', 'wb') as f:
    pickle.dump(data, f)
print("✅ Data saved to: results/improved/data/improved_data.pkl")

# Create plots
print("\nCreating visualizations...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Attendance
axes[0].plot(episode_attendances, alpha=0.6, label='Attendance', linewidth=0.8)
axes[0].axhline(y=60, color='r', linestyle='--', linewidth=2, label='Optimal (60)')
axes[0].fill_between(range(len(episode_attendances)), 55, 65, 
                     alpha=0.2, color='green', label='Target range (±5)')

# Moving average
window = 100
if len(episode_attendances) >= window:
    moving_avg = np.convolve(episode_attendances, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(episode_attendances)), moving_avg, 
                'orange', linewidth=2, label=f'{window}-Episode Moving Avg')

axes[0].set_xlabel('Episode', fontsize=11)
axes[0].set_ylabel('Attendance', fontsize=11)
axes[0].set_title('Improved Baseline - Bar Attendance', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-5, 105)

# Plot 2: Rewards
axes[1].plot(episode_rewards, alpha=0.6, label='Episode Reward', linewidth=0.8)
if len(episode_rewards) >= window:
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    axes[1].plot(range(window-1, len(episode_rewards)), moving_avg, 
                'r-', linewidth=2, label=f'{window}-Episode Moving Avg')

axes[1].set_xlabel('Episode', fontsize=11)
axes[1].set_ylabel('Total Reward', fontsize=11)
axes[1].set_title('Training Rewards', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Plot 3: Epsilon
axes[2].plot(epsilon_history, linewidth=2, color='purple')
axes[2].axhline(y=0.1, color='gray', linestyle=':', linewidth=2, label='Exploration threshold')
axes[2].set_xlabel('Episode', fontsize=11)
axes[2].set_ylabel('Epsilon', fontsize=11)
axes[2].set_title('Exploration Rate (Slower Decay: 0.9995)', fontsize=13, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('results/improved/figures/improved_training.png', dpi=150)
print("✅ Training plot saved to: results/improved/figures/improved_training.png")

# Statistics
print()
print("="*70)
print("FINAL STATISTICS")
print("="*70)

# Last 200 episodes (converged behavior)
final_attendances = episode_attendances[-200:]
mean_att = np.mean(final_attendances)
std_att = np.std(final_attendances)
median_att = np.median(final_attendances)

distances = np.abs(np.array(final_attendances) - 60)
mae = np.mean(distances)
within_5 = 100 * np.sum(distances <= 5) / len(distances)
within_10 = 100 * np.sum(distances <= 10) / len(distances)

print(f"Last 200 Episodes Performance:")
print(f"  Mean Attendance:        {mean_att:.2f}")
print(f"  Std Deviation:          {std_att:.2f}")
print(f"  Median:                 {median_att:.1f}")
print(f"  Mean Abs Error:         {mae:.2f}")
print(f"  % within ±5:            {within_5:.1f}%")
print(f"  % within ±10:           {within_10:.1f}%")
print()
print(f"Final Epsilon:            {agent.epsilon:.4f}")
print()

# Comparison with baseline (from your previous results)
print("="*70)
print("COMPARISON WITH BASELINE")
print("="*70)
print("                    Baseline (0.995)    Improved (0.9995)")
print("-"*70)
print(f"Mean Attendance:         60.72              {mean_att:.2f}")
print(f"Std Deviation:           29.45              {std_att:.2f}")
print(f"% within ±5:             0.0%               {within_5:.1f}%")
print(f"% within ±10:            10.5%              {within_10:.1f}%")
print(f"Epsilon at ep 2000:      0.010              {agent.epsilon:.4f}")
print("="*70)

# Copy to desktop
os.system('cp results/improved/figures/improved_training.png /mnt/c/Users/Utilisateur/Desktop/ 2>/dev/null')
print("\n✅ Plot copied to Windows Desktop!")

print()
print("="*70)
print("ALL DONE! 🎉")
print("="*70)
print("Next steps:")
print("1. Check results/improved/figures/improved_training.png")
print("2. Compare with baseline results")
print("3. Analyze if coordination improved")
print("="*70)
