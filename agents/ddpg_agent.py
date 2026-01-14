"""
Deep Deterministic Policy Gradient (DDPG) Agent
For continuous action spaces (bar pricing)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class ActorNetwork(nn.Module):
    """
    Actor Network (Policy)
    Maps state to action (price)
    """
    
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        
        self.action_low = action_low
        self.action_high = action_high
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, state):
        """
        Forward pass
        Output is scaled to [action_low, action_high]
        """
        normalized = self.network(state)
        # Scale from [0,1] to [action_low, action_high]
        action = self.action_low + normalized * (self.action_high - self.action_low)
        return action


class CriticNetwork(nn.Module):
    """
    Critic Network (Q-function)
    Maps (state, action) to Q-value
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        # Q(s,a) network
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """
        Forward pass
        Concatenate state and action, output Q-value
        """
        x = torch.cat([state, action], dim=1)
        q_value = self.network(x)
        return q_value


class OUNoise:
    """
    Ornstein-Uhlenbeck process for exploration noise
    Generates temporally correlated noise
    """
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """Generate noise sample"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ReplayBuffer:
    """Experience replay buffer for DDPG"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent
    For continuous action spaces (e.g., pricing)
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        action_low,
        action_high,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        noise_std=0.2
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_low: Minimum action value (e.g., min price = 2.0)
            action_high: Maximum action value (e.g., max price = 15.0)
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Replay buffer capacity
            noise_std: Standard deviation for exploration noise
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        
        # Actor networks (policy)
        self.actor = ActorNetwork(state_dim, action_dim, action_low, action_high)
        self.actor_target = ActorNetwork(state_dim, action_dim, action_low, action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks (Q-function)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Exploration noise
        self.noise = OUNoise(action_dim, sigma=noise_std)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state, add_noise=True):
        """
        Select action given state
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
        
        Returns:
            action: Selected action (numpy array)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Add exploration noise
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            # Clip to valid range
            action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=64):
        """
        Update actor and critic networks
        
        Returns:
            critic_loss, actor_loss: Losses for logging
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # =============== Update Critic ===============
        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        # Current Q-value
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # =============== Update Actor ===============
        # Actor loss: maximize Q(s, actor(s))
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # =============== Soft Update Target Networks ===============
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source, target):
        """
        Soft update of target network
        target = tau * source + (1 - tau) * target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


if __name__ == "__main__":
    print("Testing DDPG Agent...")
    
    # Create agent
    agent = DDPGAgent(
        state_dim=12,      # Bar state dimension
        action_dim=1,      # Price (single continuous action)
        action_low=2.0,    # Min price
        action_high=15.0   # Max price
    )
    
    print("Agent created successfully!")
    print(f"  State dim: {agent.state_dim}")
    print(f"  Action dim: {agent.action_dim}")
    print(f"  Action range: [{agent.action_low}, {agent.action_high}]")
    
    # Test action selection
    dummy_state = np.random.rand(12)
    action = agent.select_action(dummy_state, add_noise=True)
    print(f"\nTest action selection:")
    print(f"  State shape: {dummy_state.shape}")
    print(f"  Selected action (price): ${action[0]:.2f}")
    
    # Test storing and updating
    print("\nTest training loop:")
    for i in range(200):
        state = np.random.rand(12)
        action = agent.select_action(state, add_noise=True)
        reward = np.random.randn() * 10  # Random reward
        next_state = np.random.rand(12)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        
        if i >= 64:  # Start updating after 64 samples
            critic_loss, actor_loss = agent.update(batch_size=32)
            
            if i % 50 == 0:
                print(f"  Step {i}: Critic Loss={critic_loss:.4f}, Actor Loss={actor_loss:.4f}")
    
    print("\n✅ DDPG Agent test successful!")
