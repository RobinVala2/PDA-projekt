import random
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import time

parser = argparse.ArgumentParser(description="Run FrozenLake policy search with specified configurations.")
parser.add_argument("--map_size", type=str, choices=["4x4", "8x8"], default="4x4",
                    help="Specify the map size for FrozenLake (4x4 or 8x8).")
args = parser.parse_args()

env = gym.make("FrozenLake-v1", map_name=args.map_size, is_slippery=False, render_mode="rgb_array")

num_states = env.observation_space.n
num_actions = env.action_space.n
num_episodes = 10 if args.map_size == "4x4" else 50 
max_steps_per_episode = 50 if args.map_size == "4x4" else 200 

solution_found = False
policy_attempts = 0

start_time = time.time()

while not solution_found:
    policy = tuple(random.choice(range(num_actions)) for _ in range(num_states))
    policy_attempts += 1
    print(f"Evaluating random policy {policy_attempts}: {policy}")
    success = False
    steps_to_goal = max_steps_per_episode + 1

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = policy[state]
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            steps += 1

            if reward == 1:
                success = True
                steps_to_goal = min(steps_to_goal, steps)
                break

        if success:
            break

    if success:
        elapsed_time = time.time() - start_time
        print(f"Successful policy found after {policy_attempts} attempts: {policy} - Steps to goal: {steps_to_goal}")
        print(f"Time taken to find the solution: {elapsed_time:.2f} seconds")  # Display elapsed time
        solution_found = True

        state = env.reset()[0]
        frames = []
        done = False
        step_count = 0
        while not done and step_count < max_steps_per_episode:
            frames.append(env.render())
            action = policy[state]
            state, reward, done, _, _ = env.step(action)
            step_count += 1
        frames.append(env.render())
        env.close()

        fig = plt.figure()
        plt.axis("off")
        im = plt.imshow(frames[0])

        def update(frame):
            im.set_data(frame)
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=300, repeat=False)
        filename = f"frozenlake_solution_{args.map_size}_steps_{steps_to_goal}.gif"
        ani.save(filename, writer="imagemagick")
        print(f"Animation saved as {filename}")
