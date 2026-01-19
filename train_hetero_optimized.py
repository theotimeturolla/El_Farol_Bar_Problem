"""Version optimisée du training heterogeneous"""
import sys
sys.path.append(".")
import numpy as np
import torch
from tqdm import tqdm
from environment.multi_bars_env import MultiBarsEnv
from agents.ddpg_agent import DDPGAgent
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import os

print("="*70)
print("TRAINING HETEROGENEOUS - VERSION OPTIMISÉE")
print("="*70)

env = MultiBarsEnv(n_consumers=180, n_bars=3)

# Agents
print("Creating agents...")
bar_agents = [DDPGAgent(12, 1, 2.0, 15.0, actor_lr=1e-3, critic_lr=1e-3) for _ in range(3)]
consumer_agent = DQNAgent(6, 4, lr=0.001)

# Epsilon heterogene FIXE
consumer_epsilons = np.random.uniform(0.15, 0.25, size=180)
print(f"Mean epsilon: {consumer_epsilons.mean():.3f}")
print()

attendances = [[], [], []]
prices = [[], [], []]

print("Starting training (200 episodes)...")
for episode in tqdm(range(200), desc="Training"):
    observations, _ = env.reset()
    
    for step in range(100):
        actions = {}
        
        # Bars
        for i, bar_id in enumerate(env.bar_agents):
            price = bar_agents[i].select_action(observations[bar_id], add_noise=True)
            actions[bar_id] = price
        
        # Consumers with individual epsilon
        for idx, cons_id in enumerate(env.consumer_agents):
            if np.random.random() < consumer_epsilons[idx]:
                choice = np.random.randint(0, 4)
            else:
                obs_t = torch.FloatTensor(observations[cons_id]).unsqueeze(0)
                with torch.no_grad():
                    q = consumer_agent.q_network(obs_t)
                choice = q.argmax().item()
            actions[cons_id] = choice
        
        next_obs, rewards, dones, truncs, infos = env.step(actions)
        
        # Store transitions
        for i, bar_id in enumerate(env.bar_agents):
            bar_agents[i].store_transition(
                observations[bar_id], actions[bar_id], 
                rewards[bar_id], next_obs[bar_id], dones[bar_id]
            )
        
        for cons_id in env.consumer_agents:
            consumer_agent.store_transition(
                observations[cons_id], actions[cons_id],
                rewards[cons_id], next_obs[cons_id], dones[cons_id]
            )
        
        # Update (every 10 steps)
        if step % 10 == 0:
            for i in range(3):
                bar_agents[i].update(batch_size=32)
            consumer_agent.update(batch_size=32)
        
        observations = next_obs
        
        if all(dones.values()):
            break
    
    # Track
    for i in range(3):
        if env.bars[i].attendance_history:
            attendances[i].append(env.bars[i].attendance_history[-1])
            prices[i].append(env.bars[i].price)
    
    # Log every 50 episodes
    if episode % 50 == 0 and episode > 0:
        print(f"\nEpisode {episode}/200")
        for i in range(3):
            if len(attendances[i]) >= 50:
                avg_att = np.mean(attendances[i][-50:])
                avg_price = np.mean(prices[i][-50:])
                print(f"  Bar {i}: Price=${avg_price:.2f}, Attendance={avg_att:.1f}")
        
        total = sum([np.mean(attendances[i][-50:]) for i in range(3) if len(attendances[i]) >= 50])
        print(f"  TOTAL: {total:.1f}/180")

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)

# Final stats
print("\nFINAL RESULTS (last 50 episodes):")
for i in range(3):
    avg_att = np.mean(attendances[i][-50:])
    avg_price = np.mean(prices[i][-50:])
    print(f"  Bar {i}: Price=${avg_price:.2f}, Attendance={avg_att:.1f}")

total = sum([np.mean(attendances[i][-50:]) for i in range(3)])
print(f"  TOTAL: {total:.1f}/180 ({100*total/180:.1f}%)")

# Save models
os.makedirs("results/hetero_quick/models", exist_ok=True)
for i in range(3):
    bar_agents[i].save(f"results/hetero_quick/models/bar_{i}_ddpg.pt")
consumer_agent.save("results/hetero_quick/models/consumer_dqn.pt")
np.save("results/hetero_quick/models/consumer_epsilons.npy", consumer_epsilons)
print("\n✅ Models saved")

# Plot
os.makedirs("results/hetero_quick/figures", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Multi-Bars Training - Heterogeneous Consumers", fontsize=14, fontweight="bold")

colors = ["blue", "green", "red"]
for i in range(3):
    axes[0].plot(attendances[i], label=f"Bar {i}", color=colors[i], alpha=0.7)
    axes[1].plot(prices[i], label=f"Bar {i}", color=colors[i], alpha=0.7)

axes[0].axhline(y=60, color="black", linestyle="--", linewidth=2, label="Optimal (60)")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Attendance")
axes[0].set_title("Bar Attendance")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Price ($)")
axes[1].set_title("Bar Prices")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/hetero_quick/figures/training.png", dpi=150)
print("✅ Plot saved to results/hetero_quick/figures/training.png")

os.system("cp results/hetero_quick/figures/training.png /mnt/c/Users/Utilisateur/Desktop/hetero_final.png 2>/dev/null")
print("✅ Copied to Desktop: hetero_final.png")

print("\n" + "="*70)
print("ALL DONE! 🎉")
print("="*70)
