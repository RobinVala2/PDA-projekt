import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from collections import deque
import imageio


def environment_setup(map_name, is_slippery, render_mode=None):
    environment = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
    return environment


def initialize_q_table(state_space, action_space):
    q_table = np.zeros((state_space, action_space))
    return q_table


def load_q_table(map_name, path):
    with open(f"{path}/frozen_lake{map_name}.pkl", "rb") as f:
        return pickle.load(f)


def choose_action(epsilon, action_space, q_table, state):
    if np.random.random() < epsilon:  # Exploration
        return action_space.sample()
    else:  # Exploitation
        return np.argmax(q_table[state, :])


def apply_penalty(new_state, step, penalty_settings, action_info, action, difference_indexes, previous_state, environment):
    reward = 0

    # Penalty for step into the hole
    if environment.unwrapped.desc.flatten()[new_state] == b'H':
        reward -= penalty_settings['hole_penalty']

    # Slippery is not an AGENT fault, do not penalize when occur
    if step not in difference_indexes:  # AGENT executed intended ACTION

        # PROBLEMATIC, HOW EFFECTIVELY AND NOT TO KILL THE RL ALGORITHM ...
        # Penalties for repeating actions (cyclical behaviour)
        #if len(action_info) > 10 and action == action_info[-1] == action_info[-3]:  # -1 ... last record
        #   reward -= penalty_settings['cyclic_action_penalty']

        # Penalties for steps used beyond treshold
        if step >= penalty_settings['step_limit']:
            reward -= penalty_settings['step_penalty']

        # Penalty for attempting to move off the map
        if new_state == previous_state:
            reward -= penalty_settings['out_of_bounds_penalty']

    return reward


# Min-Max normalization due penalization (negative Q-table values)
def normalize_q_table(q_table):
    # Find the minimum and maximum values in the Q-table
    min_q = np.min(q_table)
    max_q = np.max(q_table)
    # Maximum value equals minimum, return a Q-table with zero values
    if max_q == min_q:
        return np.zeros_like(q_table)
    # Normalize the Q-table to the range [0, 1]
    normalized_q_table = (q_table - min_q) / (max_q - min_q)
    return normalized_q_table


# Identify the executed slippery ACTION
def classify_move(new_state, previous_state, action):
    move = new_state - previous_state  # Get value of tiles moved by AGENT
    if move in (-4, -8):
        return 3  # up
    elif move == 1:
        return 2  # right
    elif move in (4, 8):
        return 1  # down
    elif move == -1:
        return 0  # left
    elif move == 0:
        return int(action)  # agent expected (generated) action


# Compare two lists of ACTIONs
def find_list_differences(action_info, real_moves_due_to_slip):
    if len(action_info) != len(real_moves_due_to_slip):
        raise ValueError("Both lists must be of the same length.")
    # Find the indexes where differences occur
    difference_indexes = [index for index, (a, b) in enumerate(zip(action_info, real_moves_due_to_slip)) if a != b]
    difference_count = len(difference_indexes)

    return difference_count, difference_indexes


def run_agent(episodes, training, map_name, is_slippery):
    output_folder = f"Q-table_solutions_{map_name}_{is_slippery}_penalization"

    render_mode = "rgb_array" if not training else None

    # Initialize environment
    environment = environment_setup(map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
    state_space = environment.observation_space.n
    action_space = environment.action_space

    if training:
        # Folder setup for saving the Q-tables for successful episodes that reach the goal 'reward = 1'
        # If folder already exists, the content is deleted

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            for file in os.listdir(output_folder):
                file_path = os.path.join(output_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        q_table = initialize_q_table(state_space, action_space.n)

        # Q-learning parameters
        learning_rate = 0.8
        discount_factor = 0.95
        epsilon = 1
        min_epsilon = 0.01
        epsilon_decay = 0.0001
        max_steps_per_episode = 100

        episode_rewards = np.zeros(episodes)
        episode_steps = np.zeros(episodes)
        episode_rating = np.zeros(episodes)
        ep_slip = np.zeros(episodes)

        # Penalty parameters
        penalty_settings = {
            'revisit_penalty': 0.005,
            'recent_visit_penalty': 0.005,
            'step_penalty': 0.001,
            'out_of_bounds_penalty': 4,
            'cyclic_action_penalty': 0.005,
            'step_limit': 20,
            'hole_penalty': 4,
        }

        for episode in range(episodes):
            epsilon = max(epsilon - epsilon_decay, min_epsilon)
            state = environment.reset()[0]

            done = False
            truncated = False

            # Penalty structures
            visited_states = set()
            last_states = deque(maxlen=6)  # Keep track of the last 6 STATEs

            # Slippery structures
            slip_infos = []  # Store info about occurring slips
            action_info = []  # Store AGENT intended ACTIONs
            real_moves_due_to_slip = []  # Store executed ACTIONs towards ENVIRONMENT

            for step in range(max_steps_per_episode):
                action = choose_action(epsilon, action_space, q_table, state)
                previous_state = state  # Store the initial STATE before the ACTION (step)
                action_info.append(int(action))  # Store the AGENT intended ACTIONs
                new_state, reward, done, truncated, _ = environment.step(action)

                # Move classification
                classified_move = classify_move(new_state, previous_state, action)
                real_moves_due_to_slip.append(classified_move)

                # Slippery index calculation
                difference_count, difference_indexes = find_list_differences(action_info, real_moves_due_to_slip)

                # Check for slippery
                if is_slippery and step in difference_indexes:
                    slip_info = f"Slip detected: Episode {episode}, Step {step}, Previous State: {previous_state}, New State: {new_state}, Action Taken: {action}"
                    slip_infos.append(slip_info)

                goal_state = environment.unwrapped.desc.flatten().tolist().index(b'G')

                # Apply penalties
                penalty = apply_penalty(new_state, step, penalty_settings, action_info, action, difference_indexes, previous_state, environment)
                reward += penalty

                q_table[state][action] = q_table[state][action] + learning_rate * (
                        reward + discount_factor * np.max(q_table[new_state]) - q_table[state][action])

                # Update state tracking
                visited_states.add(state)
                last_states.append(state)
                state = new_state

                if done or truncated:
                    break

            episode_steps[episode] = step
            episode_rating[episode] = reward
            ep_slip[episode] = int(len(slip_infos))

            if done and environment.unwrapped.desc.flatten()[state] == b'G':
                reward = 1.0

            if reward == 1:
                # Print slippery results
                for info in slip_infos:
                    print(info)

                print(f"Agent expected action: {action_info}, step count: {len(action_info)}.")
                print(f"Agent executed action due slippery: {real_moves_due_to_slip}, slip count: {difference_count}. ")
                print(f"Success! Reached the goal in episode {episode}.", flush=True)
                print("Current Q-table: \n" + np.array2string(q_table,
                                                              formatter={'float_kind': lambda x: f"{x:.10f}"}) + "\n")
                episode_rewards[episode] = 1

                # Normalize Q-table, get rid of negative values
                # Sigmoid not that effective (slow, demanding, ...)
                q_table = normalize_q_table(q_table)

                q_table_filename = os.path.join(output_folder, f"Q-table-episode-{episode}.txt")
                with open(q_table_filename, "w") as f:
                    f.write(np.array2string(q_table, formatter={'float_kind': lambda x: f"{x:.10f}"}))

        environment.close()

        sum_rewards = np.zeros(episodes)

        for episode in range(episodes):
            sum_rewards[episode] = np.sum(episode_rewards[max(0, episode - 100):(episode + 1)])

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(episode_rating)
        plt.title("Rating per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.plot(episode_steps)
        plt.title("Steps per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.plot(ep_slip)
        plt.title("Slips per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Slips")
        plt.grid()

        plt.tight_layout()
        plt.savefig(f'{output_folder}/frozen_lake_episode{map_name}.png')
        plt.show()

        f = open(f"{output_folder}/frozen_lake{map_name}.pkl", "wb")
        pickle.dump(q_table, f)
        f.close()

        last_q_table = os.path.join(output_folder, "last_q_table.txt")
        with open(last_q_table, "w") as f:
            f.write(np.array2string(q_table, formatter={'float_kind': lambda x: f"{x:.10f}"}))

        return q_table, episode_rewards

    elif not training:
        q_table = load_q_table(map_name, output_folder)
        q_table = normalize_q_table(q_table)

        print("Current Q-table:")
        print(np.array2string(q_table, formatter={'float_kind': lambda x: f"{x:.10f}"}))

        successful_episodes = 0
        best_steps = float("inf")

        # Identify the goal state
        goal_state = environment.unwrapped.desc.flatten().tolist().index(b'G')

        for episode in range(episodes):
            # Reset the environment for each episode
            state = environment.reset()[0]
            done = False
            frames = []
            steps = 0

            # Capture the initial frame
            initial_frame = environment.render()
            if isinstance(initial_frame, np.ndarray) and initial_frame.ndim == 3:
                frames.append(initial_frame)

            while not done:
                # Exploit the learned Q-table (greedy policy)
                action = np.argmax(q_table[state, :])
                new_state, reward, done, truncated, _ = environment.step(action)
                frame = environment.render()

                # Capture each frame for GIF generation
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    frames.append(frame)

                state = new_state
                steps += 1

            # Check if the agent reached the goal state
            if state == goal_state:
                successful_episodes += 1

                # Save only if this is the fastest solution so far
                if steps < best_steps:
                    best_steps = steps
                    gif_filename = os.path.join(output_folder, f"solution_episode_{episode}_steps_{steps}.gif")
                    with imageio.get_writer(gif_filename, mode="I", duration=1) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                    print(f"Saved best solution as GIF: {gif_filename}")
                else:
                    print(f"Skipped episode {episode} with number of steps: {steps}")
            else:
                print(f"Episode {episode} ended without reaching the goal. Steps: {steps}")

        # Display results
        success_rate = (successful_episodes / episodes) * 100
        print("\nResults of Evaluation:")
        print(f"  Total Episodes: {episodes}")
        print(f"  Successful Episodes: {successful_episodes}")
        print(f"  Success Rate: {success_rate:.2f}%")

        environment.close()
        return q_table, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Frozen Lake Q-learning agent.")

    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training or testing')
    parser.add_argument('--training', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Training mode: True or False (default: True)')
    parser.add_argument('--map_name', type=str, default='4x4',
                        help='Map name for the FrozenLake environment, 8x8 or 4x4')
    parser.add_argument('--is_slippery', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Set whether the environment is slippery: True or False (default: True)')

    args = parser.parse_args()

    q_table, training_rewards = run_agent(args.episodes, args.training, args.map_name, args.is_slippery)

    '''
        How to run:
        Train the agent on 8x8 map, slippery True: python q_table_run-penalization.py --episodes 10000 --training True --map_name "8x8" --is_slippery True
        Show the results of the training on 8x8 map, slippery True:  python q_table_run-penalization.py --episodes 10000 --training False --map_name "8x8" --is_slippery True
        Train the agent on 4x4 map, slippery True: python q_table_run-penalization.py --episodes 10000 --training True --map_name "4x4" --is_slippery True
        Show the results of the training on 4x4 map, slippery True:  python q_table_run-penalization.py --episodes 10000 --training False --map_name "4x4" --is_slippery True
    '''