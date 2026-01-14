"""
El Farol Bar Environment
Classic coordination problem from Arthur (1994)
"""
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces


class ElFarolEnv(ParallelEnv):
    """
    El Farol Bar environment with N consumers deciding whether to go to the bar.
    
    The bar is enjoyable with ~60 people, but unpleasant if too empty or too crowded.
    """
    
    metadata = {"name": "el_farol_v0"}
    
    def __init__(self, n_consumers=100, optimal_attendance=60, render_mode=None):
        """
        Args:
            n_consumers: Number of consumer agents
            optimal_attendance: Optimal number of people at the bar
            render_mode: Not used for now
        """
        super().__init__()
        
        self.n_consumers = n_consumers
        self.optimal_attendance = optimal_attendance
        self.render_mode = render_mode
        
        # Define agents
        self.possible_agents = [f"consumer_{i}" for i in range(n_consumers)]
        
        # Observation: history of last 5 attendance numbers
        self.observation_space = spaces.Box(
            low=0, high=n_consumers, shape=(5,), dtype=np.float32
        )
        
        # Action: 0 = stay home, 1 = go to bar
        self.action_space = spaces.Discrete(2)
        
        # State
        self.timestep = 0
        self.attendance_history = []
        self.max_timesteps = 100
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.timestep = 0
        self.attendance_history = [self.optimal_attendance] * 5  # Initialize with optimal
        
        observations = {agent: self._get_observation() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """
        Execute one step in the environment.
        
        Args:
            actions: Dict of {agent_id: action} where action is 0 or 1
        
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Calculate attendance
        attendance = sum(actions.values())
        self.attendance_history.append(attendance)
        
        # Calculate rewards
        rewards = {}
        for agent_id in self.agents:
            if actions[agent_id] == 1:  # Went to the bar
                # Reward based on how close attendance is to optimal
                distance_from_optimal = abs(attendance - self.optimal_attendance)
                reward = 10.0 - (distance_from_optimal / 10.0)  # Max reward: 10, decreases with distance
            else:  # Stayed home
                reward = 5.0  # Base reward for staying home
            
            rewards[agent_id] = reward
        
        # Get new observations
        observations = {agent: self._get_observation() for agent in self.agents}
        
        # Check if episode is done
        self.timestep += 1
        terminated = self.timestep >= self.max_timesteps
        
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"attendance": attendance} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_observation(self):
        """
        Get observation: last 5 attendance numbers
        """
        history = self.attendance_history[-5:]
        # Normalize to [0, 1]
        history_normalized = np.array(history, dtype=np.float32) / self.n_consumers
        return history_normalized
    
    def render(self):
        """Render the environment (optional)"""
        if self.render_mode == "human":
            if self.attendance_history:
                print(f"Timestep {self.timestep}: Attendance = {self.attendance_history[-1]}")
    
    def close(self):
        """Clean up resources"""
        pass


if __name__ == "__main__":
    # Test the environment
    print("Testing El Farol Environment...")
    
    env = ElFarolEnv(n_consumers=100, optimal_attendance=60)
    observations, infos = env.reset(seed=42)
    
    print(f"Number of agents: {len(env.agents)}")
    print(f"Observation shape: {observations['consumer_0'].shape}")
    print(f"Sample observation: {observations['consumer_0']}")
    
    # Run a few random steps
    for step in range(5):
        # Random actions
        actions = {agent: np.random.randint(2) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        attendance = infos['consumer_0']['attendance']
        avg_reward = np.mean(list(rewards.values()))
        
        print(f"Step {step+1}: Attendance = {attendance}, Avg Reward = {avg_reward:.2f}")
    
    print("\n Environment test successful!")
