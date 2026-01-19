"""
Multi-Bars with HETEROGENEOUS consumers (different epsilon per agent)
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


def train_multi_bars_heterogeneous(
    n_episodes=1000,
    n_consumers=180,
    n_bars=3,
    batch_size=64,
    save_dir="results/multi_bars_hetero"
):
    """Train with HETEROGENEOUS consumers"""
    
    print("="*70)
    print("MULTI-BARS - HETEROGENEOUS CONSUMERS")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"Consumers: {n_consumers}")
    print(f"Bars: {n_bars}")
    print()
    print("KEY INNOVATION:")
    print("  ✅ Each consumer has INDIVIDUAL epsilon")
    print("  ✅ Diversity in exploration → No mass effect")
    print("  ✅ Always some agents exploring, some exploiting")
    print("="*70)
    print()
    
    # Create environment
    env = MultiBarsEnv(n_consumers=n_consumers, n_bars=n_bars)
    
    # Create bar agents
    print("Creating DDPG agents for bars...")
    bar_agents = []
    for i in range(n_bars):
        agent = DDPGAgent(
            state_dim=12,
            action_dim=1,
            action_low=2.0,
            action_high=15.0,
            actor_lr=1e-4,
            critic_lr=1e-3,
            noise_std=0.5
        )
        bar_agents.append(agent)
    print(f"✅ {n_bars} DDPG agents created")
    
    # Create ONE shared DQN agent
    print("Creating shared DQN agent...")
    consumer_state_dim = n_bars * 2
    consumer_action_dim = n_bars + 1
    shared_agent = DQNAgent(
        state_dim=consumer_state_dim,
        action_dim=consumer_action_dim,
        lr=0.001,
        epsilon_decay=1.0,
        epsilon_start=1.0,
        epsilon_end=1.0
    )
    
    # HETEROGENEOUS epsilon per consumer (FIXED)
    consumer_epsilons = np.random.uniform(0.05, 0.30, size=n_consumers)
    
    print(f"✅ Shared DQN with heterogeneous exploration:")
    print(f"   Epsilon range: {consumer_epsilons.min():.3f} - {consumer_epsilons.max():.3f}")
    print(f"   Mean epsilon: {consumer_epsilons.mean():.3f}")
    print()
    
    # Tracking
    episode_rewards_bars = [[] for _ in range(n_bars)]
    episode_rewards_consumers = []
    episode_attendances = [[] for _ in range(n_bars)]
    episode_prices = [[] for _ in range(n_bars)]
    
    # Training loop
    import torch
    print("Starting training...")
    for episode in tqdm(range(n_episodes), desc="Training"):
        observations, _ = env.reset()
        
        episode_reward_bars = [0] * n_bars
        episode_reward_consumers = 0
        
        for step in range(env.max_timesteps):
            actions = {}
            
            # Bars select prices
            for i, bar_agent_id in enumerate(env.bar_agents):
                obs = observations[bar_agent_id]
                price = bar_agents[i].select_action(obs, add_noise=True)
                actions[bar_agent_id] = price
            
            # Consumers with INDIVIDUAL epsilon
            for idx, consumer_id in enumerate(env.consumer_agents):
                obs = observations[consumer_id]
                
                if np.random.random() < consumer_epsilons[idx]:
                    # Explore
                    bar_choice = np.random.randint(0, consumer_action_dim)
                else:
                    # Exploit
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    with torch.no_grad():
                        q_values = shared_agent.q_network(obs_tensor)
                    bar_choice = q_values.argmax().item()
                
                actions[consumer_id] = bar_choice
            
            # Step
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store transitions
            for i, bar_agent_id in enumerate(env.bar_agents):
                bar_agents[i].store_transition(
                    observations[bar_agent_id],
                    actions[bar_agent_id],
                    rewards[bar_agent_id],
                    next_observations[bar_agent_id],
                    terminations[bar_agent_id]
                )
            
            for consumer_id in env.consumer_agents:
                shared_agent.store_transition(
                    observations[consumer_id],
                    actions[consumer_id],
                    rewards[consumer_id],
                    next_observations[consumer_id],
                    terminations[consumer_id]
                )
            
            # Update
            for i in range(n_bars):
                bar_agents[i].update(batch_size=batch_size)
            
            shared_agent.update(batch_size=batch_size)
            
            # Track
            for i, bar_agent_id in enumerate(env.bar_agents):
                episode_reward_bars[i] += rewards[bar_agent_id]
            
            episode_reward_consumers += sum([rewards[c] for c in env.consumer_agents])
            
            observations = next_observations
            
            if all(terminations.values()):
                break
        
        # Update target
        if episode % 10 == 0:
            shared_agent.update_target_network()
        
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
            print(f"Mean Epsilon: {consumer_epsilons.mean():.3f}")
            
            for i in range(n_bars):
                avg_att = np.mean(episode_attendances[i][-100:])
                avg_price = np.mean(episode_prices[i][-100:])
                avg_rew = np.mean(episode_rewards_bars[i][-100:])
                print(f"Bar {i}: Price=${avg_price:.2f}, Att={avg_att:.1f}, Rew={avg_rew:.2f}")
            
            total = sum([np.mean(episode_attendances[i][-100:]) for i in range(n_bars)])
            print(f"TOTAL: {total:.1f} / {n_consumers}")
            print(f"{'='*70}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    
    # Save
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    for i in range(n_bars):
        bar_agents[i].save(f"{save_dir}/models/bar_{i}_ddpg.pt")
    shared_agent.save(f"{save_dir}/models/consumer_dqn.pt")
    np.save(f"{save_dir}/models/consumer_epsilons.npy", consumer_epsilons)
    print("✅ Models saved")
    
    # Plot
    from experiments.train_multi_bars_fixed import plot_results
    plot_results(episode_rewards_bars, episode_attendances, episode_prices,
                 episode_rewards_consumers, n_consumers, save_dir)
    
    return bar_agents, shared_agent


if __name__ == "__main__":
    train_multi_bars_heterogeneous(n_episodes=1000, n_consumers=180, n_bars=3)
