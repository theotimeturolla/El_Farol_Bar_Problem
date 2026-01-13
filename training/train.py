"""
Training script for El Farol Bar baseline
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment.el_farol_env import ElFarolEnv
from agents.dqn_agent import DQNAgent


def train_baseline(
    n_episodes=1000,
    n_consumers=100,
    optimal_attendance=60,
    batch_size=64,
    target_update_freq=10,
    save_dir="results"
):
    """Train baseline El Farol model with DQN agents"""
    
    print("=" * 60)
    print("Training El Farol Bar - Baseline")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Consumers: {n_consumers}")
    print(f"Optimal attendance: {optimal_attendance}")
    print()
    
    # Create environment
    env = ElFarolEnv(n_consumers=n_consumers, optimal_attendance=optimal_attendance)
    
    # Create agent (parameter sharing - all consumers use same agent)
    state_dim = 5
    action_dim = 2
    agent = DQNAgent(state_dim, action_dim)
    
    # Tracking
    episode_rewards = []
    episode_attendances = []
    episode_losses = []
    
    # Training loop
    for episode in tqdm(range(n_episodes), desc="Training"):
        observations, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(env.max_timesteps):
            # Select actions for all agents (using same policy)
            actions = {}
            for agent_id in env.agents:
                obs = observations[agent_id]
                action = agent.select_action(obs, training=True)
                actions[agent_id] = action
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store transitions for all agents
            for agent_id in env.agents:
                agent.store_transition(
                    observations[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    next_observations[agent_id],
                    terminations[agent_id]
                )
            
            # Update agent
            loss = agent.update(batch_size=batch_size)
            if loss > 0:
                episode_loss.append(loss)
            
            episode_reward += sum(rewards.values())
            observations = next_observations
            
            if all(terminations.values()):
                break
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Track metrics
        final_attendance = env.attendance_history[-1]
        episode_rewards.append(episode_reward)
        episode_attendances.append(final_attendance)
        if episode_loss:
            episode_losses.append(np.mean(episode_loss))
        
        # Logging
        if episode % 50 == 0 and episode > 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_attendance = np.mean(episode_attendances[-50:])
            print(f"\nEpisode {episode}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Attendance: {avg_attendance:.1f} (optimal: {optimal_attendance})")
            print(f"  Epsilon: {agent.epsilon:.3f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Save model
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    agent.save(f"{save_dir}/models/baseline_dqn.pt")
    print(f"Model saved to {save_dir}/models/baseline_dqn.pt")
    
    # Plot results
    plot_training_results(episode_rewards, episode_attendances, optimal_attendance, save_dir)
    
    return agent, episode_rewards, episode_attendances


def plot_training_results(rewards, attendances, optimal, save_dir="results"):
    """Plot training results"""
    
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    axes[0].plot(rewards, alpha=0.6, label='Episode Reward')
    # Moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot attendance
    axes[1].plot(attendances, alpha=0.6, label='Attendance')
    axes[1].axhline(y=optimal, color='r', linestyle='--', linewidth=2, label=f'Optimal ({optimal})')
    # Moving average
    if len(attendances) >= window:
        moving_avg = np.convolve(attendances, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(attendances)), moving_avg, 'g-', linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Attendance')
    axes[1].set_title('Bar Attendance Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/baseline_training.png", dpi=150)
    print(f"Training plots saved to {save_dir}/figures/baseline_training.png")
    plt.close()


if __name__ == "__main__":
    train_baseline(n_episodes=1000)
