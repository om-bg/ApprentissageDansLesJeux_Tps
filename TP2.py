import gymnasium as gym
import numpy as np
import random
import time

env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()

print(f"Actions space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

q_table = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.1  
gamma = 0.99  
epsilon = 1.0  
epsilon_decay = 0.995 
epsilon_min = 0.01  
num_episodes = 5000  

def choose_action(state):
    if random.random() < epsilon:
        action = env.action_space.sample()  
    else:
        action = np.argmax(q_table[state])  
    return action

# Entraînement de l'agent
print("Début de l'entraînement...")
successes_train = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        # Mise à jour de la Q-table
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error
        
        state = next_state
        
        if done and reward > 0:
            successes_train += 1
    
    # Décroissance de epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    
    # Affichage des progrès
    if (episode + 1) % 100 == 0:
        success_rate = (successes_train / (episode + 1)) * 100
        print(f"Épisode {episode + 1}: Taux de réussite = {success_rate:.2f}%")

# Test de l'agent
print("\nTest de l'agent entraîné...")
num_test_episodes = 10
successes_test = 0

for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        time.sleep(0.5)  
    
    if reward > 0:
        successes_test += 1
    print(f"Test {episode + 1}: Récompense = {episode_reward}")

success_rate = (successes_test / num_test_episodes) * 100
print(f"\nTaux de réussite final: {success_rate}%")
print("Q-table finale (arrondie):\n", np.round(q_table, 2))
env.close()