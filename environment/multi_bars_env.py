"""
Multi-Bar Environment
Multiple bars competing for customers
"""
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
from bar import Bar


class MultiBarsEnv(ParallelEnv):
    """
    Environment with multiple bars and consumers
    - Bars set prices (continuous action)
    - Consumers choose which bar to go to (discrete action)
    """
    
    metadata = {"name": "multi_bars_v0"}
    
    def __init__(self, n_consumers=100, n_bars=3, render_mode=None):
        super().__init__()
        
        self.n_consumers = n_consumers
        self.n_bars = n_bars
        self.render_mode = render_mode
        
        # Create bars
        self.bars = [Bar(bar_id=i, capacity_optimal=60) for i in range(n_bars)]
        
        # Define agents
        self.bar_agents = [f"bar_{i}" for i in range(n_bars)]
        self.consumer_agents = [f"consumer_{i}" for i in range(n_consumers)]
        self.possible_agents = self.bar_agents + self.consumer_agents
        
        # State
        self.timestep = 0
        self.max_timesteps = 100
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.timestep = 0
        
        # Reset bars
        for bar in self.bars:
            bar.reset()
        
        observations = {}
        
        # Bar observations
        for i, bar_agent in enumerate(self.bar_agents):
            observations[bar_agent] = self.bars[i].get_state()
        
        # Consumer observations
        for consumer_agent in self.consumer_agents:
            observations[consumer_agent] = self._get_consumer_observation()
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """
        Execute one timestep
        actions: {
            "bar_0": [price],
            "bar_1": [price],
            ...
            "consumer_0": choice (0=stay home, 1=bar_0, 2=bar_1, ...),
            ...
        }
        """
        # 1. Bars set prices
        for i, bar_agent in enumerate(self.bar_agents):
            price = actions[bar_agent]
            if isinstance(price, np.ndarray):
                price = price[0]
            self.bars[i].set_price(price)
        
        # 2. Count attendance at each bar
        bar_attendance = [0] * self.n_bars
        consumer_choices = {}
        
        for consumer_agent in self.consumer_agents:
            choice = actions[consumer_agent]
            consumer_choices[consumer_agent] = choice
            
            if choice > 0:  # Not staying home
                bar_idx = choice - 1
                if bar_idx < self.n_bars:
                    bar_attendance[bar_idx] += 1
        
        # 3. Calculate rewards
        rewards = {}
        
        # Bar rewards
        for i, bar_agent in enumerate(self.bar_agents):
            attendance = bar_attendance[i]
            reward = self.bars[i].calculate_reward(attendance)
            rewards[bar_agent] = reward
        
        # Consumer rewards
        for consumer_agent in self.consumer_agents:
            choice = consumer_choices[consumer_agent]
            
            if choice == 0:  # Stayed home
                rewards[consumer_agent] = 5.0
            else:
                bar_idx = choice - 1
                if bar_idx < self.n_bars:
                    bar = self.bars[bar_idx]
                    attendance = bar_attendance[bar_idx]
                    
                    # Utility = enjoyment - price paid
                    crowding_penalty = abs(attendance - bar.capacity_optimal) / 10.0
                    enjoyment = 10.0 - crowding_penalty
                    utility = enjoyment - bar.price
                    
                    rewards[consumer_agent] = utility
                else:
                    rewards[consumer_agent] = 0.0
        
        # 4. New observations
        observations = {}
        
        for i, bar_agent in enumerate(self.bar_agents):
            observations[bar_agent] = self.bars[i].get_state()
        
        for consumer_agent in self.consumer_agents:
            observations[consumer_agent] = self._get_consumer_observation()
        
        # 5. Termination
        self.timestep += 1
        terminated = self.timestep >= self.max_timesteps
        
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        infos = {agent: {"bar_attendance": bar_attendance} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_consumer_observation(self):
        """
        Consumer observes:
        - Current price of each bar (normalized)
        - Recent average attendance of each bar (normalized)
        """
        prices = []
        attendances = []
        
        for bar in self.bars:
            # Price normalized to [0, 1]
            price_norm = (bar.price - 2.0) / (15.0 - 2.0)
            prices.append(price_norm)
            
            # Average recent attendance
            if len(bar.attendance_history) > 0:
                recent_att = np.mean(bar.attendance_history[-3:])
            else:
                recent_att = bar.capacity_optimal
            
            att_norm = recent_att / 100.0
            attendances.append(att_norm)
        
        obs = np.array(prices + attendances, dtype=np.float32)
        return obs
    
    def render(self):
        if self.render_mode == "human":
            print(f"\nTimestep {self.timestep}:")
            for i, bar in enumerate(self.bars):
                if bar.attendance_history:
                    att = bar.attendance_history[-1]
                    print(f"  Bar {i}: Price={bar.price:.2f}, Attendance={att}")
    
    def close(self):
        pass


if __name__ == "__main__":
    print("Testing Multi-Bars Environment...")
    
    env = MultiBarsEnv(n_consumers=100, n_bars=3)
    observations, _ = env.reset(seed=42)
    
    print(f"Number of bar agents: {len(env.bar_agents)}")
    print(f"Number of consumer agents: {len(env.consumer_agents)}")
    print(f"Bar observation shape: {observations['bar_0'].shape}")
    print(f"Consumer observation shape: {observations['consumer_0'].shape}")
    
    # Run a few steps
    for step in range(3):
        actions = {}
        
        # Bars set random prices
        for bar_agent in env.bar_agents:
            actions[bar_agent] = np.random.uniform(3.0, 8.0, size=(1,))
        
        # Consumers make random choices
        for consumer_agent in env.consumer_agents:
            actions[consumer_agent] = np.random.randint(0, 4)  # 0-3
        
        observations, rewards, dones, truncs, infos = env.step(actions)
        
        print(f"\nStep {step+1}:")
        print(f"  Bar attendance: {infos['bar_0']['bar_attendance']}")
        print(f"  Bar 0 reward: {rewards['bar_0']:.2f}")
        print(f"  Avg consumer reward: {np.mean([rewards[c] for c in env.consumer_agents]):.2f}")
    
    print("\n Multi-Bars Environment test successful!")
