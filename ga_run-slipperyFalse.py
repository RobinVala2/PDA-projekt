import argparse
import os
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool

# Environment setup
def setup_environment(map_name="4x4", is_slippery=False):
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", map_name=map_name, is_slippery=is_slippery)
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    return env, num_actions, num_states

# Random policy
def create_random_policy(num_states, num_actions):
    return np.random.randint(num_actions, size=num_states)

# Run episode
def run_simulation(env, policy, max_steps):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    visited_states = set()

    for step in range(max_steps):
        obs = int(obs)
        action = policy[obs]
        obs, reward, done, truncated, _ = env.step(action)
        steps += 1

        # Goal reached
        if done and reward == 1:
            total_reward += 1.0  # Maximum reward for reaching the goal
            break

        # Exploration reward
        if obs not in visited_states:
            total_reward += 0.1  # Reward for new state
            visited_states.add(obs)

        # Step penalty
        total_reward -= 0.01

    # Ensure reward is in range [0, 1]
    total_reward = max(0, min(1, total_reward))
    return total_reward, steps

# Parallelized fitness evaluation
def calculate_fitness_parallel(population, map_name, is_slippery, max_steps):
    with Pool(processes=4) as pool:
        scores = pool.map(
            calculate_single_policy_fitness,
            [(policy, map_name, is_slippery, max_steps) for policy in population],
        )
    return scores

def calculate_single_policy_fitness(args):
    policy, map_name, is_slippery, max_steps = args
    env, _, _ = setup_environment(map_name=map_name, is_slippery=is_slippery)
    return calculate_fitness(env, policy, max_steps)

def calculate_fitness(env, policy, max_steps):
    total_reward = 0
    for _ in range(10):  # Average over multiple episodes
        reward, _ = run_simulation(env, policy, max_steps)
        total_reward += reward
    return total_reward / 10  # Average fitness

# Fitness proportional selection
def fitness_proportional_selection(population, scores, num_select):
    scores = np.array(scores)
    min_score = np.min(scores)

    # Shift scores to be non-negative
    if min_score < 0:
        scores += abs(min_score) + 1e-6

    total_score = np.sum(scores)

    # Handle case where all scores are zero
    if total_score == 0:
        # Assigns equal probability to all individuals (uniform probabilities), ensuring no bias in selection.
        probabilities = np.ones(len(scores)) / len(scores)  
    else:
        probabilities = scores / total_score

    # Ensure probabilities are valid for selection
    # len(probabilities[probabilities > 0]): Counts the number of individuals with non-zero probabilities.
    if len(probabilities[probabilities > 0]) < num_select:
        # random selection is used with equal probabilities (replace=False prevents duplicates).
        selected_indices = np.random.choice(len(population), size=num_select, replace=False)
    else:
        # individuals are selected using the computed probabilities
        selected_indices = np.random.choice(len(population), size=num_select, p=probabilities, replace=False)

    return [population[i] for i in selected_indices]

# Crossover
def apply_crossover(selected_policies):
    offspring = []
    for i in range(0, len(selected_policies) - 1, 2):
        crossover_point = random.randint(1, len(selected_policies[0]) - 1)
        parent1 = selected_policies[i]
        parent2 = selected_policies[i + 1]
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        offspring.extend([child1, child2])
    return offspring


# Mutation
def apply_mutation(policies, mutation_rate, num_actions):
    for policy in policies:
        if np.random.rand() < mutation_rate:
            mutation_index = np.random.randint(len(policy))
            policy[mutation_index] = np.random.randint(num_actions)
    return policies

# Render GIF
def create_gif(env, policy, title, filename, path):
    frames = []
    obs, _ = env.reset()
    done = False

    while not done:
        img = env.render()
        frames.append(img)
        action = policy[int(obs)]
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    if reward == 1:
        frames.append(env.render())
        for i in range(len(frames)):
            img = Image.fromarray(frames[i])
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()

            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (img.width - text_width) // 2
            draw.text((text_x, 10), title, font=font, fill=(255, 255, 255))
            frames[i] = np.array(img)


        filepath = os.path.join(f'{path}', filename)
        imageio.mimsave(filepath, frames, fps=10)
        print(f"Saved GIF: {filepath}")
    else:
        print(f"Skipping GIF for {title}, goal not reached.")

# Main GA loop
def run_evolution(map_name="4x4", is_slippery=False, pop_size=200, num_generations=100, max_steps=25):
    ga_directory = f'GA_solutions_slippery_False_{map_name}'
    if os.path.exists(ga_directory):
        for file_name in os.listdir(ga_directory):
            os.unlink(os.path.join(ga_directory, file_name))
    else:
        os.makedirs(ga_directory)
    
    env, num_actions, num_states = setup_environment(map_name=map_name, is_slippery=is_slippery)
    population = [create_random_policy(num_states, num_actions) for _ in range(pop_size)]
    mutation_rate = 0.2

    # Track scores and average steps for plotting
    best_scores = []
    average_rewards = []
    average_steps = []

    for generation in range(num_generations):
        print(f"Generation: {generation + 1}")

        # Calculate fitness and track steps
        total_steps = []
        scores = calculate_fitness_parallel(population, map_name, is_slippery, max_steps)
        for policy in population:
            _, steps = run_simulation(env, policy, max_steps)
            total_steps.append(steps)

        best_score = max(scores)
        avg_score = np.mean(scores)
        avg_steps = np.mean(total_steps)  # Track average steps

        best_scores.append(best_score)
        average_rewards.append(avg_score)
        average_steps.append(avg_steps)

        print(f"Best Score: {best_score:.4f}, Average Score: {avg_score:.4f}, Average Steps: {avg_steps:.2f}")

        # Select top performers
        selected = fitness_proportional_selection(population, scores, pop_size // 2)

        # Apply crossover and mutation
        offspring = apply_crossover(selected)
        offspring = apply_mutation(offspring, mutation_rate, num_actions)

        # Add random exploration (every generation)
        population += [create_random_policy(num_states, num_actions) for _ in range(pop_size // 5)]

        # Combine elites with offspring
        elite_count = max(1, int(0.1 * pop_size))           # calculates the number of elites
        elite_indices = np.argsort(scores)[-elite_count:]   # identifies the indices of the top-performing individuals
        elites = [population[i] for i in elite_indices]     # retrieves the elite individuals from the population
        population = elites + offspring

        # Ensure population size
        population = population[:pop_size]

    # Save successful policies as GIFs
    successful_policies = []
    for i, policy in enumerate(population):
        reward, steps = run_simulation(env, policy, max_steps)  # Unpack reward and steps
        if reward >= 1.0:  # Threshold to identify successful policies (goal reached)
            successful_policies.append(policy)
            create_gif(env, policy, f'Solution {i + 1}', f'solution_{i + 1}.gif', ga_directory)

    if not successful_policies:
        print("No successful policies found!")
    else:
        print(f"Saved {len(successful_policies)} successful policies.")

    # Plotting scores and average steps
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(best_scores, label='Best Scores')
    plt.plot(average_rewards, label='Average Scores', color='orange')
    plt.xlabel('Generations')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(average_steps, label='Average Steps per Generation', color='green')
    plt.xlabel('Generations')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'{ga_directory}/evolution_performance.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm on FrozenLake Environment, Slippery True")
    parser.add_argument(
        "--map_size", type=str, default="4x4", choices=["4x4", "8x8"],
        help="Map size for the FrozenLake environment (default: 4x4)"
    )
    args = parser.parse_args()

    if args.map_size == "4x4":
        population = 200
        generation = 200
        steps = 50
    elif args.map_size == "8x8":
        population = 300
        generation = 300
        steps = 100
    
    run_evolution(map_name=args.map_size, is_slippery=False, pop_size=population, num_generations=generation, max_steps=steps)
   
