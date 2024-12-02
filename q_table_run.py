import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
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
    if np.random.random() < epsilon:                # Exploration
        return action_space.sample()
    else:                                           # Exploitation
        return np.argmax(q_table[state, :])

def run_agent(episodes, training, map_name, is_slippery):

    output_folder = f"Q-table_solutions_{map_name}_{is_slippery}"
    
    render_mode = "rgb_array" if not training else None

    # initialize environment
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

        # Parametes
        learning_rate = 0.8                 # how much new information overrides old information during Q-value update
        discount_factor = 0.95              # how much importance is given to future rewards, 'gamma'
        epsilon = 1                         # initial value, exploration probability
        min_epsilon = 0.01                  # minimum value of epsilon
        epsilon_decay = 0.0001              # factor by which epsilon decreases after each episode
        max_steps_per_episode = 200         # how long can agent run per episode

        episode_steps = np.zeros(episodes)
        episode_rating = np.zeros(episodes)
        episode_rewards = np.zeros(episodes)

        for episode in range(episodes):

            epsilon = max(epsilon - epsilon_decay, min_epsilon) # linear decay
            state = environment.reset()[0]      # returning to initial state
            done = False
            truncated = False
            episode_reward = 0
            step_count = 0

            for step in range(max_steps_per_episode):

                action = choose_action(epsilon, action_space, q_table, state)
                new_state, reward, done, truncated, _ = environment.step(action)        # apply the selected action
                q_table[state][action] = q_table[state][action] + learning_rate * (
                                reward + discount_factor * np.max(q_table[new_state]) - q_table[state][action])

                state = new_state
                episode_reward += reward
                step_count += 1

                if done or truncated:
                    break
            
            episode_rating[episode] = episode_reward
            episode_steps[episode] = step_count

            if epsilon < 0.1:
                learning_rate = 0.1 * learning_rate

            if reward == 1:
                print(f"Success! Reached the goal in episode {episode}.", flush=True)
                print("Current Q-table: \n" + np.array2string(q_table, formatter={'float_kind': lambda x: f"{x:.10f}"}))
                episode_rewards[episode] = 1

        environment.close()

        sum_rewards = np.zeros(episodes)

        for episode in range(episodes):
            sum_rewards[episode] = np.sum(episode_rewards[max(0, episode - 100):(episode + 1)])

        # Plot reward per episode
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(range(episodes), episode_rating, label="Rewards per Episode")
        plt.title("Rating per Episode") 
        plt.xlabel("Episode") 
        plt.ylabel("Rewards")
        plt.grid()
        plt.legend()

        # Plot steps per episode
        plt.subplot(3, 1, 2)
        plt.plot(range(episodes), episode_steps, label="Steps per Episode", color='orange')
        plt.title("Steps per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid()
        plt.legend()

        # Plot cumulative success over episodes
        plt.subplot(3, 1, 3)
        cumulative_rewards = np.cumsum(episode_rewards)
        plt.plot(range(episodes), cumulative_rewards, label="Cumulative Success", color='green')
        plt.title("Cumulative Success Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Success")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output_folder}/frozen_lake_episode{map_name}.png')
        plt.show()
       
        # save the Q-table
        f = open(f"{output_folder}/frozen_lake{map_name}.pkl", "wb")
        pickle.dump(q_table, f)
        f.close()

        # save the last Q-table
        last_q_table = os.path.join(output_folder, "last_q_table.txt")
        with open(last_q_table, "w") as f:
            f.write(np.array2string(q_table, formatter={'float_kind': lambda x: f"{x:.10f}"}))

        return q_table, episode_rewards
    
    elif not training:
        
        q_table = load_q_table(map_name, output_folder)
        print("Current Q-table:")
        print(np.array2string(q_table, formatter={'float_kind': lambda x: f"{x:.10f}"}))

        # Evaluation: Calculate success rate
        successful_episodes = 0
        best_steps = float("inf")

        for episode in range(episodes):
            state = environment.reset()[0]
            done = False
            frames = []
            steps = 0

            initial_frame = environment.render()
            if isinstance(initial_frame, np.ndarray) and initial_frame.ndim == 3:
                frames.append(initial_frame)

            while not done:
                action = np.argmax(q_table[state, :])  # Exploit learned policy
                new_state, reward, done, truncated, _ = environment.step(action)
                frame = environment.render()
                if isinstance(frame, np.ndarray) and frame.ndim ==3:
                    frames.append(frame)
                state = new_state
                steps += 1

            if reward > 0:  # If the goal is reached
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

    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes for training or testing')
    parser.add_argument('--training', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Training mode: True or False (default: True)')
    parser.add_argument('--map_name', type=str, default='8x8',
                        help='Map name for the FrozenLake environment, 8x8 or 4x4')
    parser.add_argument('--is_slippery', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Set whether the environment is slippery: True or False (default: True)')

    args = parser.parse_args()

    q_table, training_rewards = run_agent(args.episodes, args.training, args.map_name, args.is_slippery)

    '''
        How to run:
        Train the agent on 8x8 map, slippery True: python q_table_run.py --episodes 15000 --training True --map_name "8x8" --is_slippery True
        Show the results of the training on 8x8 map, slippery True:  python q_table_run.py --episodes 1000 --training False --map_name "8x8" --is_slippery True
        Train the agent on 8x8 map, slippery False: python q_table_run.py --episodes 15000 --training True --map_name "8x8" --is_slippery False
        Show the results of the training on 8x8 map, slippery False:  python q_table_run.py --episodes 1000 --training False --map_name "8x8" --is_slippery False
        Train the agent on 4x4 map, slippery True: python q_table_run.py --episodes 15000 --training True --map_name "4x4" --is_slippery True
        Show the results of the training on 4x4 map, slippery True:  python q_table_run.py --episodes 1000 --training False --map_name "4x4" --is_slippery True
        Train the agent on 4x4 map, slippery False: python q_table_run.py --episodes 15000 --training True --map_name "4x4" --is_slippery False
        Show the results of the training on 4x4 map, slippery False:  python q_table_run.py --episodes 1000 --training False --map_name "4x4" --is_slippery False
    
    '''