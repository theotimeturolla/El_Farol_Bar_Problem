"""Test rapide heterogeneous - version simplifiée"""
import sys
sys.path.append('.')
import numpy as np
from tqdm import tqdm
from environment.multi_bars_env import MultiBarsEnv
from agents.ddpg_agent import DDPGAgent
from agents.dqn_agent import DQNAgent
import torch

print("TEST RAPIDE - 5 épisodes")

env = MultiBarsEnv(n_consumers=180, n_bars=3)

# Bar agents
bar_agents = [DDPGAgent(12, 1, 2.0, 15.0) for _ in range(3)]

# Consumer agent
consumer_agent = DQNAgent(6, 4, lr=0.001)

# HETEROGENEOUS epsilon
consumer_epsilons = np.random.uniform(0.1, 0.3, size=180)
print(f"Mean epsilon: {consumer_epsilons.mean():.3f}")

for episode in range(5):
    print(f"\nEpisode {episode+1}/5")
    observations, _ = env.reset()
    
    total_att = [0, 0, 0]
    
    for step in range(100):  # 100 timesteps per episode
        actions = {}
        
        # Bars (rapide)
        for i, bar_id in enumerate(env.bar_agents):
            price = bar_agents[i].select_action(observations[bar_id], add_noise=False)
            actions[bar_id] = price
        
        # Consumers (avec epsilon individuel)
        for idx, cons_id in enumerate(env.consumer_agents):
            if np.random.random() < consumer_epsilons[idx]:
                choice = np.random.randint(0, 4)  # explore
            else:
                obs_t = torch.FloatTensor(observations[cons_id]).unsqueeze(0)
                with torch.no_grad():
                    q = consumer_agent.q_network(obs_t)
                choice = q.argmax().item()
            
            actions[cons_id] = choice
        
        # Step
        observations, rewards, dones, truncs, infos = env.step(actions)
        
        # PAS DE TRAINING pour le test (c'est ça qui est lent)
        
        if all(dones.values()):
            break
    
    # Stats
    for i in range(3):
        if env.bars[i].attendance_history:
            att = env.bars[i].attendance_history[-1]
            total_att[i] = att
            print(f"  Bar {i}: Price=${env.bars[i].price:.2f}, Attendance={att}")
    
    print(f"  TOTAL: {sum(total_att)}/180")

print("\n✅ Test terminé en quelques secondes!")
