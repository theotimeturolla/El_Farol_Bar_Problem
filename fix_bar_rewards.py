# Modifier environment/bar.py pour que les bars apprennent mieux

import re

with open('environment/bar.py', 'r') as f:
    content = f.read()

# Trouver la fonction calculate_reward
old_reward = r"def calculate_reward\(self, attendance\):.*?return reward"

new_reward = '''def calculate_reward(self, attendance):
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
        
        return reward'''

content = re.sub(old_reward, new_reward, content, flags=re.DOTALL)

with open('environment/bar.py', 'w') as f:
    f.write(content)

print("✅ Bar rewards improved:")
print("   - STRONG penalty for empty bar (-100)")
print("   - Incentive to attract customers")
print("   - Bars will learn to lower prices to get customers")
