import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
import matplotlib.pyplot as plt

#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################


def decode_state(state: int):
    a_r = state // 1000
    a_c = (state // 100) % 10
    c_r = (state // 10) % 10
    c_c = state % 10
    return int(a_r), int(a_c), int(c_r), int(c_c)


def manhattan(a_r, a_c, b_r, b_c):
    return abs(a_r - b_r) + abs(a_c - b_c)


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
    episodes = 5000  # Training is capped at 5000 episodes for this project

    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################

    learning_rate = 0.5
    discount_factor = 0.99

    epsilon = 1
    min_epsilon = 0.01
    epsilon_decay_rate = 0.002

    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    ep_steps = []

    step_penalty = -1
    catch_reward = 100
    distance_bonus = 1.2
    max_steps_per_episode = 30
    wall_penalty = 5

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

        last_positions = []

        while not terminated and not truncated and steps < max_steps_per_episode:

            # epsilon-greedy (pure greedy + exploration)
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
                # q = q_table[state]
                # prob = np.exp(q) / np.sum(np.exp(q))
                # action = int(rng.choice(len(q), p=prob))

            # decode old state
            a_r, a_c, c_r, c_c = decode_state(state)

            prev_dist = manhattan(a_r, a_c, c_r, c_c)

            # take action
            new_state, _, terminated, truncated, _ = env.step(action)

            # decode new state
            na_r, na_c, nc_r, nc_c = decode_state(new_state)
            new_dist = manhattan(na_r, na_c, nc_r, nc_c)

            # reward shaping
            if na_r == nc_r and na_c == nc_c:
                reward = catch_reward
                print(f"[DEBUG] Step {steps}: Caught the cat! Reward = {reward}")
            else:
                reward = step_penalty + ((prev_dist - new_dist) * distance_bonus)
                if new_dist > prev_dist:
                    reward -= 2
                if (na_r, na_c) in last_positions:
                    reward -= 0.5
                if na_r < 0 or na_r > 7 or na_c < 0 or na_c > 7:
                    reward -= wall_penalty
                    print(f"[DEBUG] Step {steps}: Hit a wall! Reward = {reward}")

            # Q-learning update
            max_q = 0 if terminated or truncated else np.max(q_table[new_state])
            old_q = q_table[state][action]
            q_table[state][action] = old_q + learning_rate * (
                reward + discount_factor * max_q - old_q
            )
            q_table[state][action] = np.clip(q_table[state][action], -500, 500)

            state = new_state
            total_reward += reward
            steps += 1

            # prevent oscillation
            last_positions.append((na_r, na_c))
            if len(last_positions) > 4:
                last_positions.pop(0)
            if last_positions.count((na_r, na_c)) > 2:
                break

        ep_steps.append(steps)
        rewards_per_episode[ep - 1] = total_reward
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay_rate * ep)

    env.close()

    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################

    # plt.figure(figsize=(10, 4))
    # plt.plot(rewards_per_episode)
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title(f"{cat_name}: Rewards per Episode")
    # plt.grid(True)
    # plt.show()

    # cat_r, cat_c = 5, 5
    # q_max_grid = np.zeros((10, 10))

    # for a_r in range(10):
    #     for a_c in range(10):
    #         state = a_r * 1000 + a_c * 100 + cat_r * 10 + cat_c
    #         q_max_grid[a_r, a_c] = np.max(q_table[state])

    # plt.figure(figsize=(6, 5))
    # plt.imshow(q_max_grid, cmap="hot", origin="lower")
    # plt.colorbar(label="Max Q-value")
    # plt.title(f"{cat_name}: Max Q-values (Cat at {cat_r},{cat_c})")
    # plt.xlabel("Agent Column")
    # plt.ylabel("Agent Row")
    # plt.show()

    # action_to_arrow = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    # policy_grid = np.empty((10, 10), dtype=str)

    # for a_r in range(10):
    #     for a_c in range(10):
    #         state = a_r * 1000 + a_c * 100 + cat_r * 10 + cat_c
    #         best_action = np.argmax(q_table[state])
    #         policy_grid[a_r, a_c] = action_to_arrow[best_action]

    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.set_xticks(np.arange(0.5, 10.5, 1))
    # ax.set_yticks(np.arange(0.5, 10.5, 1))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.grid(True)

    # for i in range(10):
    #     for j in range(10):
    #         ax.text(j, i, policy_grid[i, j], ha="center", va="center", fontsize=16)

    # ax.set_title(f"{cat_name}: Policy Map (Cat at {cat_r},{cat_c})")
    # plt.show()

    # If rendering is enabled, play an episode every 'render' episodes
    if render != -1 and (ep == 1 or ep % render == 0):
        viz_env = make_env(cat_type=cat_name)
        play_q_table(
            viz_env,
            q_table,
            max_steps=100,
            move_delay=0.02,
            window_title=f"{cat_name}: Training Episode {ep}/{episodes}",
        )
        print("episode", ep)

    return q_table
