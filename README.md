# ApprentissageDansLesJeux_Tps

Learning the fundamentals of Reinforcement Learning. With the used of the famous **<font color="#ee1313">OpenAI Gym</font>** library.

| TP                               | Environment                                | Algorithm/Technique                                              |
| -------------------------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| <font color="#245bdb">1st</font> | <font color="#ffff00">CartPole-v1</font>   | <font color="#33ddff">Manual interaction (basic RL tools)</font> |
| <font color="#ff0000">2nd</font> | <font color="#ffff00">FrozenLake-v1</font> | <font color="#33ddff">Q-learning</font>                          |
| <font color="#00b050">3rd</font> | <font color="#ffff00">Taxi-v3</font>       | <font color="#33ddff">Policy-based learning & Q-learning</font>  |



---

## TP1 CartPole-v1 :

In this TP, we worked with CartPole-v1 to understand the basic of Reinforcement Learning environments.

### Installation of OpenAI Gym

```bash
pip install --upgrade gymnasium pygame
```

### Environment Initialization

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
```

    

---

## TP2 FrozenLake-v1 (Q-Learning) :

In this TP, we implemented **Q-learning**, a Reinforcement Learning algorithm.

### Key Steps:

1. **Initialize the environment:**
    
    ```python
    env = gym.make("FrozenLake-v1", is_slippery=False)
    ```
    
2. **Create a Q-table:**
    
    ```python
    import numpy as np
    
    q_table = np.zeros((16, 4))
    ```
    
3. **Understanding the algorithm:**
    
    - **Exploration vs. Exploitation:** The agent initially explores the environment randomly and then starts choosing optimal actions based on learned values.
        
    - **Q-value update:** Using the Bellman equation:
        
        ```python
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        ```
        
4. **Training the agent:**
    
    - Running 5000 episodes to update the Q-table.
        
    - Using an **epsilon-greedy strategy** where the agent chooses random actions with probability `epsilon`, decreasing over time.
        
5. **Testing the trained agent:**
    
    - Running test episodes where the agent chooses the best-known action based on the Q-table.
        
    - Measuring the **success rate** to evaluate performance.
        

---

## TP3 Taxi-v3 Policy-based Learning :

In this TP, we implemented policy-based learning methods alongside Q-learning.

### Key Concepts:

1. **Understanding the environment:**
    
    - The agent must pick up a passenger and drop them off at the correct location.
        
    - **Action space:** 6 discrete actions (move in four directions, pick up, drop off).
        
    - **State space:** 500 possible states, representing taxi position, passenger location, and destination.
        
    - **Reward structure:**
        
        - +20 for a successful drop-off.
            
        - -1 for each step taken.
            
        - -10 for illegal pickups/drop-offs.
            
2. **Creating a Policy Table and Value Table:**
    
    ```python
    import numpy as np
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    policy_table = np.ones((state_size, action_size)) / action_size
    value_table = np.zeros(state_size)
    ```
    
3. **Training the agent using a policy iteration approach:**
    
    - Selecting actions based on policy.
        
    - Updating value functions based on the observed rewards.
        
4. **Q-learning Implementation:**
    
    - **Exploration-exploitation trade-off:** Using an **epsilon-greedy strategy**.
        
    - **Q-value update using the Bellman equation:**
        
        ```python
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        ```
        
    - **Hyperparameters:**
        
        ```python
        alpha = 0.1  # Learning rate
        gamma = 0.99  # Discount factor
        epsilon = 1.0  # Exploration rate
        epsilon_decay = 0.999  # Decay factor for exploration
        ```
        
5. **Testing and Evaluating the Agent:**
    
    - Running test episodes where the agent selects the best-known action.
        
    - Calculating the **success rate** to measure performance.
        
    - **Success is defined as successfully dropping off the passenger.**
        
