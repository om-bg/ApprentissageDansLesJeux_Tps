import random
import time

GRID_SIZE = 5
TRAPS = [(1, 1), (1, 3)] 
TREASURE = (3, 2)  
START_POS = (0, 0)  
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
memory = {}

def reset_game():
    return START_POS

def move(position, action):
    x, y = position
    if action == 'UP' and x > 0:
        return (x-1, y)
    elif action == 'DOWN' and x < GRID_SIZE-1:
        return (x+1, y)
    elif action == 'LEFT' and y > 0:
        return (x, y-1)
    elif action == 'RIGHT' and y < GRID_SIZE-1:
        return (x, y+1)
    return position 

def get_reward(new_pos):
    if new_pos in TRAPS:
        return -10, True  # Trap = end of game with penalty
    elif new_pos == TREASURE:
        return 10, True  # Treasure = end of game with reward
    return -1, False  # Normal move

def choose_action(position, previous_position, exploration_rate=0.3):
    possible_actions = []
    for action in ACTIONS:
        if move(position, action) != previous_position:
            possible_actions.append(action)
            
    if not possible_actions:
        possible_actions = ACTIONS

    # Exploration
    if random.random() < exploration_rate or position not in memory:
        return random.choice(possible_actions)
    
    # Exploitation: choose the best known action
    best_action = None
    best_score = -float('inf')

    for action in possible_actions:
        if action in memory.get(position, {}):
            score = memory[position][action]
            if score > best_score:
                best_score = score
                best_action = action
    
    return best_action if best_action else random.choice(possible_actions)

def update_memory(position, action, reward):
    if position not in memory:
        memory[position] = {action: reward}
    else:
        memory[position][action] = memory[position].get(action, 0) * 0.9 + reward * 0.1

def play_episode(exploration_rate=0.3):
    position = reset_game()
    previous_position = None
    total_reward = 0
    path = [position]
    done = False
    reached_treasure = False
    entered_trap = False
    
    while not done:
        action = choose_action(position, previous_position, exploration_rate)
        new_pos = move(position, action)
        reward, done = get_reward(new_pos)
        
        update_memory(position, action, reward)
        
        previous_position = position  # Update last position
        position = new_pos
        path.append(position)
        total_reward += reward
        
        # Check if the agent reached the treasure or entered a trap
        if new_pos == TREASURE:
            reached_treasure = True
        if new_pos in TRAPS:
            entered_trap = True
        
        time.sleep(0.3)
    
    return total_reward, path, reached_treasure, entered_trap

def train_agent(episodes=100):
    exploration_rate = 0.5  
    
    for episode in range(1, episodes+1):
        exploration_rate = max(0.01, exploration_rate * 0.99)
        reward, path, reached_treasure, entered_trap = play_episode(exploration_rate)
        
        # Display the result of the episode
        if reached_treasure:
            print(f"Épisode {episode}: Score = {reward}, Longueur du chemin = {len(path)-1} - Trésor atteint!")
        elif entered_trap:
            print(f"Épisode {episode}: Score = {reward}, Longueur du chemin = {len(path)-1} - Piège rencontré!")
        else:
            print(f"Épisode {episode}: Score = {reward}, Longueur du chemin = {len(path)-1} - Jeu non terminé")
        
        if episode % 20 == 0:
            print("Meilleur chemin trouvé:", path)

print("Début de l'apprentissage...")
train_agent(100)
print("\nTest final avec exploration minimale:")
final_reward, final_path, reached_treasure, entered_trap = play_episode(exploration_rate=0.01)
if reached_treasure:
    print(f"Score final: {final_reward} - Trésor atteint!")
elif entered_trap:
    print(f"Score final: {final_reward} - Piège rencontré!")
else:
    print(f"Score final: {final_reward} - Jeu non terminé")
print("Chemin optimal:", final_path)
