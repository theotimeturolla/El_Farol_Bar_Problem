"""
Bar agent that can set prices and has capacity
"""
import numpy as np


class Bar:
    """
    A bar that can strategically set prices to attract customers
    """
    
    def __init__(self, bar_id, capacity_optimal=60, base_cost=2.0, initial_price=5.0):
        """
        Args:
            bar_id: Unique identifier
            capacity_optimal: Optimal number of customers
            base_cost: Cost per customer served (marginal cost)
            initial_price: Starting price
        """
        self.bar_id = bar_id
        self.capacity_optimal = capacity_optimal
        self.base_cost = base_cost
        self.price = initial_price
        
        # History tracking
        self.attendance_history = []
        self.revenue_history = []
        self.price_history = []
        
    def set_price(self, price):
        """Set price (clipped to reasonable range)"""
        self.price = np.clip(price, 2.0, 15.0)
        self.price_history.append(self.price)
    
    def calculate_reward(self, attendance):
        """
        Calculate reward for the bar based on:
        - Revenue: attendance * (price - base_cost)
        - Strong penalty for being empty (wasted opportunity)
        - Penalty for being overcrowded
        """
        # Profit from customers
        revenue = attendance * (self.price - self.base_cost)
        
        # STRONG incentive to attract customers
        if attendance == 0:
            # No customers = huge penalty
            capacity_penalty = -100.0
        elif attendance < self.capacity_optimal * 0.3:
            # Very few customers = big penalty
            occupancy_rate = attendance / self.capacity_optimal
            capacity_penalty = -50 * (0.3 - occupancy_rate)
        elif attendance > self.capacity_optimal * 1.5:
            # Too crowded = penalty
            occupancy_rate = attendance / self.capacity_optimal
            capacity_penalty = -20 * (occupancy_rate - 1.5)
        else:
            # Good occupancy = bonus
            capacity_penalty = 10.0
        
        reward = revenue + capacity_penalty
        
        self.attendance_history.append(attendance)
        self.revenue_history.append(revenue)
        
        return reward
    
    def get_state(self):
        """
        State observable by the bar for learning pricing strategy
        """
        # Recent attendance (normalized)
        recent_attendance = self.attendance_history[-5:] if len(self.attendance_history) >= 5 else [0] * 5
        while len(recent_attendance) < 5:
            recent_attendance.insert(0, 0)
        attendance_norm = np.array(recent_attendance) / self.capacity_optimal
        
        # Recent revenue (normalized)
        recent_revenue = self.revenue_history[-5:] if len(self.revenue_history) >= 5 else [0] * 5
        while len(recent_revenue) < 5:
            recent_revenue.insert(0, 0)
        max_revenue = self.capacity_optimal * (15.0 - self.base_cost)
        revenue_norm = np.array(recent_revenue) / max_revenue
        
        # Current price (normalized)
        price_norm = (self.price - 2.0) / (15.0 - 2.0)
        
        # Capacity (normalized)
        capacity_norm = self.capacity_optimal / 100.0
        
        state = np.concatenate([
            attendance_norm,      # 5 values
            revenue_norm,         # 5 values
            [price_norm],         # 1 value
            [capacity_norm]       # 1 value
        ])
        
        return state.astype(np.float32)
    
    def reset(self):
        """Reset history at start of new episode"""
        self.attendance_history = []
        self.revenue_history = []
        self.price_history = []
        self.price = 5.0


if __name__ == "__main__":
    print("Testing Bar class...")
    
    bar = Bar(bar_id=0, capacity_optimal=60)
    
    # Simulate some timesteps
    for t in range(10):
        attendance = np.random.randint(20, 100)
        reward = bar.calculate_reward(attendance)
        
        print(f"Timestep {t}: Attendance={attendance}, Price={bar.price:.2f}, Reward={reward:.2f}")
        
        state = bar.get_state()
        
        new_price = bar.price + np.random.uniform(-1, 1)
        bar.set_price(new_price)
    
    print("\n✅ Bar class test successful!")
