"""Forced Participation + Price Competition Fix"""
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
print("FORCED PARTICIPATION + PRICE COMPETITION")
print("="*70)

class CompetitivePricingEnv(MultiBarsEnv):
    """Forced participation + penalty for high prices"""
    
    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # Consumer reward shaping (forced participation)
        for consumer_id in self.consumer_agents:
            choice = actions.get(consumer_id, 0)
            
            if choice == 0:
                consecutive_stays = getattr(self, f"{consumer_id}_stays", 0) + 1
                setattr(self, f"{consumer_id}_stays", consecutive_stays)
                penalty = -10.0 * (1.5 ** min(consecutive_stays, 5))
                rewards[consumer_id] = penalty
            else:
                setattr(self, f"{consumer_id}_stays", 0)
                bar_idx = choice - 1
                if bar_idx < self.n_bars:
                    original_reward = rewards[consumer_id]
                    rewards[consumer_id] = original_reward + 3.0
        
        # BAR reward modification: PENALIZE high prices
        for i, bar_agent_id in enumerate(self.bar_agents):
            bar = self.bars[i]
            attendance = bar.attendance_history[-1] if bar.attendance_history else 0
            
            # Original revenue
            revenue = attendance * (bar.price - bar.base_cost)
            
            # PENALTY for prices above competitive level (8 dollars)
            competitive_price = 8.0
            if bar.price > competitive_price:
                price_penalty = ((bar.price - competitive_price) ** 2) * 20.0
            else:
                # BONUS for competitive pricing
                price_penalty = -5.0
            
            # PENALTY for empty bar (encourage attracting customers)
            if attendance < 30:
                occupancy_penalty = -50.0
            elif attendance > 80:
                occupancy_penalty = -30.0
            else:
                occupancy_penalty = 10.0
            
            # New reward
            new_reward = revenue - price_penalty + occupancy_penalty
            rewards[bar_agent_id] = new_reward
        
        return observations, rewards, terminations, truncations, infos

env = CompetitivePricingEnv(n_consumers=180, n_bars=3)

print("REWARD MODIFICATIONS:")
print("  CONSUMERS:")
print("    - Stay home: exponential penalty")
print("    + Go to bar: bonus +3.0")
print("  BARS:")
print("    + Revenue: attendance * (price - cost)")
print("    - Price > 8 dollars: penalty (price-8)^2 * 20")
print("    + Price <= 8 dollars: bonus -5")
print("    - Attendance < 30 or > 80: penalty")
print()

print("Creating agents...")
bar_agents = [DDPGAgent(12, 1, 2.0, 15.0, actor_lr=1e-3, critic_lr=1e-3) for _ in range(3)]
consumer_agent = DQNAgent(6, 4, lr=0.005)
consumer_epsilons = np.random.uniform(0.10, 0.20, size=180)
print(f"Mean epsilon: {consumer_epsilons.mean():.3f}")
print()

attendances = [[], [], []]
prices = [[], [], []]

print("Training (300 episodes)...")
for episode in tqdm(range(300), desc="Training"):
    observations, _ = env.reset()
    
    for cons_id in env.consumer_agents:
        setattr(env, f"{cons_id}_stays", 0)
    
    for step in range(100):
        actions = {}
        
        for i, bar_id in enumerate(env.bar_agents):
            price = bar_agents[i].select_action(observations[bar_id], add_noise=True)
            actions[bar_id] = price
        
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
        
        for i, bar_id in enumerate(env.bar_agents):
            bar_agents[i].store_transition(
                observations[bar_id], actions[bar_id],
                rewards[bar_id], next_obs[bar_id], dones[bar_id]
            )
            bar_agents[i].update(batch_size=64)
        
        for cons_id in env.consumer_agents:
            consumer_agent.store_transition(
                observations[cons_id], actions[cons_id],
                rewards[cons_id], next_obs[cons_id], dones[cons_id]
            )
        
        if step % 5 == 0:
            consumer_agent.update(batch_size=64)
        
        observations = next_obs
        
        if all(dones.values()):
            break
    
    for i in range(3):
        if env.bars[i].attendance_history:
            attendances[i].append(env.bars[i].attendance_history[-1])
            prices[i].append(env.bars[i].price)
    
    if episode % 50 == 0 and episode > 0:
        print(f"\nEpisode {episode}/300")
        for i in range(3):
            if len(attendances[i]) >= 50:
                print(f"  Bar {i}: ${np.mean(prices[i][-50:]):.2f}, Att={np.mean(attendances[i][-50:]):.1f}")
        total = sum([np.mean(attendances[i][-50:]) for i in range(3) if len(attendances[i]) >= 50])
        print(f"  TOTAL: {total:.1f}/180 ({100*total/180:.1f}%)")

print("\n" + "="*70)
print("COMPLETED!")
print("="*70)

print("\nFINAL (last 50 episodes):")
for i in range(3):
    print(f"  Bar {i}: ${np.mean(prices[i][-50:]):.2f}, Att={np.mean(attendances[i][-50:]):.1f}")
total = sum([np.mean(attendances[i][-50:]) for i in range(3)])
print(f"  TOTAL: {total:.1f}/180 ({100*total/180:.1f}%)")

os.makedirs("results/competitive_pricing", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Competitive Pricing - Full Solution", fontweight="bold")
colors = ["blue", "green", "red"]
for i in range(3):
    axes[0].plot(attendances[i], label=f"Bar {i}", color=colors[i], alpha=0.7)
    axes[1].plot(prices[i], label=f"Bar {i}", color=colors[i], alpha=0.7)
axes[0].axhline(60, color="black", linestyle="--", linewidth=2, label="Optimal")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Attendance")
axes[0].set_title("Bar Attendance")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].axhline(8, color="green", linestyle="--", linewidth=2, label="Competitive (8)")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Price ($)")
axes[1].set_title("Bar Prices")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 16)
plt.tight_layout()
plt.savefig("results/competitive_pricing/training.png", dpi=150)
os.system("cp results/competitive_pricing/training.png /mnt/c/Users/Utilisateur/Desktop/competitive_pricing.png 2>/dev/null")
print("\nSaved to Desktop: competitive_pricing.png")

for i in range(3):
    bar_agents[i].save(f"results/competitive_pricing/bar_{i}.pt")
consumer_agent.save("results/competitive_pricing/consumer.pt")
np.save("results/competitive_pricing/attendances.npy", attendances)
np.save("results/competitive_pricing/prices.npy", prices)
print("Models and data saved")
