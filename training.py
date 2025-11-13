import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################








#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    # Q-learning hyperparameters
    learning_rate = 0.5     # Alpha: How much we learn from new information
    discount_factor = 0.99  # Gamma: How much we value future rewards
    
    # Epsilon-greedy strategy parameters for exploration-exploitation trade-off
    epsilon = 1.0           # Initial exploration rate (100% exploration)
    max_epsilon = 1.0       # Max exploration rate
    min_epsilon = 0.01      # Minimum exploration rate
    decay_rate = 0.001      # Rate at which epsilon decays exponentially
    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.       #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ##############################################################################        
        
        # 1. Reset the environment
        state = env.reset()
        # Handle environments that return (observation, info) tuple
        if isinstance(state, tuple):
            state = state[0]
            
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            
            # 2. Decide whether to explore or exploit (Epsilon-greedy)
            exploration_tradeoff = random.uniform(0, 1)
            
            if exploration_tradeoff < epsilon:
                # Explore: Select a random action
                action = env.action_space.sample()
            else:
                # Exploit: Select the best action from the Q-table
                action = np.argmax(q_table[state])
                
            # 3. Take the action and observe the next state and outcome
            # env.step() returns (next_state, reward, terminated, truncated, info)
            # We ignore the environment's reward (using '_') as per the hint
            next_state, _, terminated, truncated, info = env.step(action)
            
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # 4. Compute reward manually
            # This is an assumed reward structure based on typical grid-world tasks
            # - A large positive reward for winning (terminated, but not truncated)
            # - A large negative reward for losing (truncated, e.g., time ran out)
            # - A small negative reward for each step to encourage efficiency
            if terminated and not truncated:
                reward = 20  # Cat won (e.g., caught the mouse)
            elif truncated:
                reward = -10 # Cat lost (e.g., time ran out)
            else:
                reward = -1  # Step penalty
                
            # 5. Update the Q-table using the Bellman equation
            # Q(s,a) = Q(s,a) + lr * (r + gamma * max(Q(s',a')) - Q(s,a))
            
            # Get the old Q-value for the current state-action pair
            old_value = q_table[state][action]
            
            # Get the maximum Q-value for the next state (the 'greedy' part)
            next_max_q = np.max(q_table[next_state])
            
            # Calculate the new Q-value
            new_value = old_value + learning_rate * (reward + discount_factor * next_max_q - old_value)
            
            # Update the Q-table
            q_table[state][action] = new_value
            
            # Move to the next state
            state = next_state
            
        # After the episode is finished, decay epsilon
        # This shifts the agent from exploration to exploitation over time
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * ep)
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table
