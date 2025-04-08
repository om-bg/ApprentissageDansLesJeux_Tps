import gymnasium as gym
import numpy as np

env = gym.make('Taxi-v3')
state_space = env.observation_space.n
action_space = env.action_space.n

policy_table = np.ones((state_space, action_space)) / action_space
value_table = np.zeros(state_space)

learning_rate = 0.001  # Réduit le taux d'apprentissage pour éviter un apprentissage erratique
gamma = 0.99
epsilon = 0.2
epochs = 1  # Diminuez le nombre d'epochs pour éviter un overfitting
update_interval = 1000
epsilon_decay = 0.995  # Ajout d'un facteur d'exploration décroit
epsilon_min = 0.05     # Valeur minimale de epsilon

def policy(state):
    return np.random.choice(action_space, p=policy_table[state])

# Agent aléatoire 
random_rewards = []

for episode in range(20):
    state, _ = env.reset()
    total_reward = 0
    while True:
        action = env.action_space.sample()  # Action aléatoire
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
        state = next_state
    random_rewards.append(total_reward)
    print(f"Épisode {episode+1}: Récompense totale: {total_reward}")

print(f"Récompense moyenne (aléatoire): {np.mean(random_rewards):.2f}")

# Entraînement avec PPO
memory = []

for episode in range(2000):
    state, _ = env.reset()
    total_reward = 0
    while True:
        action = policy(state)  # Choisir une action selon la politique
        next_state, reward, terminated, truncated, _ = env.step(action)
        memory.append((state, action, reward, next_state, terminated or truncated))
        
        if len(memory) >= update_interval:
            # Extraction des valeurs de la mémoire
            states, actions, rewards, next_states, dones = zip(*memory)
            states = np.array(states)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            dones = np.array(dones)

            # Calcul des avantages (advantage)
            advantages = rewards + gamma * value_table[next_states] * (1 - dones) - value_table[states]
            old_probs = policy_table[states, actions]

            # Mise à jour de la politique et des valeurs
            for _ in range(epochs):
                new_probs = policy_table[states, actions]
                ratio = new_probs / old_probs
                clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
                policy_gradient = ratio * advantages
                clipped_policy_gradient = clipped_ratio * advantages

                policy_table[states, actions] += learning_rate * np.minimum(policy_gradient, clipped_policy_gradient)
                policy_table[states] = np.clip(policy_table[states], 1e-6, 1)
                policy_table[states] /= policy_table[states].sum(axis=1, keepdims=True)

            value_table[states] += learning_rate * (rewards + gamma * value_table[next_states] * (1 - dones) - value_table[states])
            memory.clear()

        state = next_state
        total_reward += reward
        if terminated or truncated:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Diminution d'epsilon au fil du temps
    print(f"Épisode {episode+1}: Récompense: {total_reward}")

# Test de l'agent
print("\n=== Test de l'agent entraîné ===")
trained_rewards = []

for episode in range(20):
    state, _ = env.reset()
    total_reward = 0
    while True:
        action = np.argmax(policy_table[state])  # Choix de l'action optimale selon la politique
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
        state = next_state
    trained_rewards.append(total_reward)
    print(f"Épisode {episode+1}: Récompense totale: {total_reward}")

# Comparaison
print("Comparaison des performances ")
print(f"Récompense moyenne avant entraînement: {np.mean(random_rewards):.2f}")
print(f"Récompense moyenne après entraînement: {np.mean(trained_rewards):.2f}")
print(f"Amélioration: {(np.mean(trained_rewards) - np.mean(random_rewards)):.2f}")

env.close()
