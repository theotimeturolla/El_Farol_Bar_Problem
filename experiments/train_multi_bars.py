"""
Training Multi-Bars Environment
3 bars with DDPG pricing + 100 consumers with DQN choice
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment.multi_bars_env import MultiBarsEnv
from agents.ddpg_agent import DDPGAgent
from agents.dqn_agent import DQNAgent


def train_multi_bars(
    n_episodes=1000,
    n_consumers=100,
    n_bars=3,
    batch_size=64,
    save_dir="results/multi_bars"
):
    """
    Train multi-bars environment
    - 3 DDPG agents for bars (pricing)
    - 1 DQN agent for consumers (choice) with parameter sharing
    """
    
    print("="*70)
    print("MULTI-BARS TRAINING")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"Number of bars: {n_bars}")
    print(f"Number of consumers: {n_consumers}")
    print()
    
    # Create environment
    env = MultiBarsEnv(n_consumers=n_consumers, n_bars=n_bars)
    
    # Create bar agents (DDPG for pricing)
    print("Creating DDPG agents for bars...")
    bar_agents = []
    for i in range(n_bars):
        agent = DDPGAgent(
            state_dim=12,      # Bar state dimension
            action_dim=1,      # Price
            action_low=2.0,    # Min price
            action_high=15.0,  # Max price
            actor_lr=1e-4,
            critic_lr=1e-3,
            noise_std=0.3      # Higher exploration for pricing
        )
        bar_agents.append(agent)
    print(f"✅ {n_bars} DDPG agents created")
    
    # Create consumer agent (DQN for bar choice)
    print("Creating DQN agent for consumers...")
    consumer_state_dim = n_bars * 2  # Prices + attendances
    consumer_action_dim = n_bars + 1  # Stay home + choose bar 0,1,2
    consumer_agent = DQNAgent(
        state_dim=consumer_state_dim,
        action_dim=consumer_action_dim,
        lr=0.001,
        epsilon_decay=0.9995
    )
    print(f"✅ DQN agent created (parameter sharing for {n_consumers} consumers)")
    print()
    
    # Tracking
    episode_rewards_bars = [[] for _ in range(n_bars)]
    episode_rewards_consumers = []
    episode_attendances = [[] for _ in range(n_bars)]
    episode_prices = [[] for _ in range(n_bars)]
    
    # Training loop
    print("Starting training...")
    for episode in tqdm(range(n_episodes), desc="Training"):
        observations, _ = env.reset()
        
        episode_reward_bars = [0] * n_bars
        episode_reward_consumers = 0
        
        for step in range(env.max_timesteps):
            actions = {}
            
            # Bars select prices (DDPG)
            for i, bar_agent_id in enumerate(env.bar_agents):
                obs = observations[bar_agent_id]
                price = bar_agents[i].select_action(obs, add_noise=True)
                actions[bar_agent_id] = price
            
            # Consumers select bars (DQN with parameter sharing)
            for consumer_id in env.consumer_agents:
                obs = observations[consumer_id]
                bar_choice = consumer_agent.select_action(obs, training=True)
                actions[consumer_id] = bar_choice
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store transitions for bar agents
            for i, bar_agent_id in enumerate(env.bar_agents):
                bar_agents[i].store_transition(
                    observations[bar_agent_id],
                    actions[bar_agent_id],
                    rewards[bar_agent_id],
                    next_observations[bar_agent_id],
                    terminations[bar_agent_id]
                )
            
            # Store transitions for consumer agent
            for consumer_id in env.consumer_agents:
                consumer_agent.store_transition(
                    observations[consumer_id],
                    actions[consumer_id],
                    rewards[consumer_id],
                    next_observations[consumer_id],
                    terminations[consumer_id]
                )
            
            # Update bar agents
            for i in range(n_bars):
                bar_agents[i].update(batch_size=batch_size)
            
            # Update consumer agent
            consumer_agent.update(batch_size=batch_size)
            
            # Track rewards
            for i, bar_agent_id in enumerate(env.bar_agents):
                episode_reward_bars[i] += rewards[bar_agent_id]
            
            episode_reward_consumers += sum([rewards[c] for c in env.consumer_agents])
            
            observations = next_observations
            
            if all(terminations.values()):
                break
        
        # Update target networks
        if episode % 10 == 0:
            consumer_agent.update_target_network()
        
        # Track metrics
        for i in range(n_bars):
            episode_rewards_bars[i].append(episode_reward_bars[i])
            episode_attendances[i].append(env.bars[i].attendance_history[-1] if env.bars[i].attendance_history else 0)
            episode_prices[i].append(env.bars[i].price)
        
        episode_rewards_consumers.append(episode_reward_consumers)
        
        # Logging
        if episode % 100 == 0 and episode > 0:
            print(f"\n{'='*70}")
            print(f"Episode {episode}/{n_episodes}")
            print(f"Consumer Agent - Epsilon: {consumer_agent.epsilon:.4f}")
            
            for i in range(n_bars):
                avg_attendance = np.mean(episode_attendances[i][-100:])
                avg_price = np.mean(episode_prices[i][-100:])
                avg_reward = np.mean(episode_rewards_bars[i][-100:])
                print(f"Bar {i}: Price=${avg_price:.2f}, Attendance={avg_attendance:.1f}, Reward={avg_reward:.2f}")
            
            avg_consumer_reward = np.mean(episode_rewards_consumers[-100:])
            print(f"Consumers: Avg Reward={avg_consumer_reward:.2f}")
            print(f"{'='*70}")
    
    print()
    print("="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    # Save models
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    for i in range(n_bars):
        bar_agents[i].save(f"{save_dir}/models/bar_{i}_ddpg.pt")
    consumer_agent.save(f"{save_dir}/models/consumer_dqn.pt")
    print(f"✅ Models saved to {save_dir}/models/")
    
    # Plot results
    plot_multi_bars_results(
        episode_rewards_bars,
        episode_attendances,
        episode_prices,
        episode_rewards_consumers,
        save_dir
    )
    
    return bar_agents, consumer_agent


def plot_multi_bars_results(rewards_bars, attendances, prices, rewards_consumers, save_dir):
    """Plot training results for multi-bars"""
    
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    n_bars = len(rewards_bars)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Multi-Bars Training Results', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Plot 1: Bar attendance
    for i in range(n_bars):
        axes[0, 0].plot(attendances[i], alpha=0.6, label=f'Bar {i}', color=colors[i])
    
    axes[0, 0].axhline(y=60, color='black', linestyle='--', linewidth=2, label='Optimal (60)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Attendance')
    axes[0, 0].set_title('Bar Attendance Over Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Bar prices
    for i in range(n_bars):
        axes[0, 1].plot(prices[i], alpha=0.6, label=f'Bar {i}', color=colors[i])
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Bar Prices Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Bar rewards
    for i in range(n_bars):
        # Moving average
        window = 50
        if len(rewards_bars[i]) >= window:
            moving_avg = np.convolve(rewards_bars[i], np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(rewards_bars[i])), moving_avg, 
                          label=f'Bar {i}', color=colors[i], linewidth=2)
    
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Reward (50-ep avg)')
    axes[1, 0].set_title('Bar Rewards (Profits)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Consumer rewards
    window = 50
    if len(rewards_consumers) >= window:
        moving_avg = np.convolve(rewards_consumers, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(rewards_consumers)), moving_avg, 
                       'purple', linewidth=2)
    
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Consumer Reward')
    axes[1, 1].set_title('Consumer Rewards (50-ep avg)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/multi_bars_training.png", dpi=150)
    print(f"✅ Training plots saved to {save_dir}/figures/multi_bars_training.png")
    
    # Copy to desktop
    os.system(f'cp {save_dir}/figures/multi_bars_training.png /mnt/c/Users/Utilisateur/Desktop/ 2>/dev/null')
    print("✅ Copied to Desktop!")
    
    plt.close()


if __name__ == "__main__":
    train_multi_bars(n_episodes=1000)
