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

def decode_state(state: int):
    a_r = state // 1000
    a_c = (state // 100) % 10
    c_r = (state // 10) % 10
    c_c = state % 10
    return int(a_r), int(a_c), int(c_r), int(c_c)

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
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    learning_rate = 0.8
    discount_factor = 0.999
    
    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay_rate = 0.0005

    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    ep_steps = []  
    
    step_penalty = -0.01
    catch_reward = 1000
    distance_bonus = 0.5
    max_steps_per_episode = 60

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        state, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0
        total_reward = 0


        while (not terminated and not truncated and steps < max_steps_per_episode):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                q = q_table[state]
                max_q = np.max(q)
                best_actions = np.where(q == max_q)[0]
                action = int(rng.choice(best_actions))

            a_r, a_c, c_r, c_c = decode_state(state)
            prev_distance = abs(a_r - c_r) + abs(a_c-c_c)
            
            new_state, _, terminated, truncated, _ = env.step(action)

            na_r, na_c, nc_r, nc_c = decode_state(new_state)
            dist_mid = abs(na_r - c_r) + abs(na_c - c_c)  
            dist_after = abs(na_r - nc_r) + abs(na_c - nc_c) 

            # compute shaped reward
            if na_r == nc_r and na_c == nc_c:
                reward = catch_reward
            else:
                reward = step_penalty

                delta = prev_distance - dist_mid
                if delta > 0:
                    reward += delta * distance_bonus

                if dist_after > dist_mid:
                    reward += step_penalty

            if terminated or truncated:
                max_q = 0
            else:
                max_q = np.max(q_table[new_state])

            old_q = q_table[state][action]
            difference = reward + discount_factor * max_q - old_q
            q_table[state][action] = old_q + learning_rate * difference

            state = new_state
            total_reward += reward
            steps += 1
        
        ep_steps.append(steps)
        rewards_per_episode[ep-1] = total_reward

        epsilon = max(min_epsilon, epsilon*np.exp(-epsilon_decay_rate*ep))

    env.close()    

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
    if render != -1 and (ep == 1 or ep % render == 0):
        viz_env = make_env(cat_type=cat_name)
        play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
        print('episode', ep)

    return q_table