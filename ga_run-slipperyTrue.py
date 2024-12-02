import os
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw, ImageFont
import argparse

# GA dictionary
# crossover, evaluate, evolution, fitness function, generation, mutation, offspring, parent, policy (strategy), population, score (in RL REWARD), selection

# GA process
# INITIALIZE (random population) - SELECTION / EVALUATION (fitness function) - CROSSOVER - MUTATION - REPEAT till termination condition met (num_generation reached)

# Set up the ENVIRONMENT
def setup_environment(map_name):        
    env = gym.make("FrozenLake-v1", render_mode="rgb_array", map_name=map_name, is_slippery=True)
    num_actions = env.action_space.n       
    num_states = env.observation_space.n    
    return env, num_actions, num_states

# Create a random POLICY (strategy) - Initialize the population
def create_random_policy(num_states, num_actions):
    # Returns array, where each it's element is a random number from a range of available ACTIONs
    return np.random.randint(num_actions, size=num_states)

# Run a game simulation with a given POLICY
def run_simulation(map_size, env, policy):
    
    if map_size == "4x4":
        max_steps = 25
    elif map_size == "8x8":
        max_steps = 100

    obs, _ = env.reset()       
    steps = 0
    for _ in range(max_steps):  
        obs = int(obs)          
        action = policy[obs]    # Get ACTION from the POLICY based on value in observation
        obs, reward, done, truncated, _ = env.step(action) 
            # obs - new AGENT STATE in the ENVIRONMENT
            # reward - if the goal is reached
            # done - game over (if goal reached or fell into a hole)
            # truncated - other game termination (max_step reached)
        steps +=1
        if done or truncated:
            return (1, steps) if done else (0, steps)   # Goal = 1; Hole, termination = 0
    return (0, steps)                                   # Return 0 if the maximum allowed steps are reached

# Calculate fitness function by running multiple simulations
def calculate_fitness(map_size, env, policy, num_simulations=10):
    results = [run_simulation(map_size, env, policy) for _ in range(num_simulations)]
    scores = [result[0] for result in results]
    steps = [result[1] for result in results]
    # Returns average from scores (population run by 10 simulations)
    raw_fitness = np.mean(scores) - 0.005 * np.mean(steps)    
    return max(0, raw_fitness)                                  # Value closer to 1 - high POLICY success (POLICY, population likely to pass, selected for crossover, mutation)
                                                                # Value closer to 0 - low POLICY success (POLICY, population likely to be forgotten)

# Evaluate all POLICies (strategies) based on their fitness function
def perform_evolution(map_size, env, population, num_elites, num_select):
    # Calculate fitness scores for all policies
    scores = [calculate_fitness(map_size, env, policy) for policy in population]

    # Sort population by fitness (highest to lowest)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_population = [population[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]

    # Select elite individuals (best policies)
    elites = sorted_population[:num_elites]

    # Perform roulette wheel selection on the rest of the population
    remaining_population = sorted_population[num_elites:]
    remaining_scores = sorted_scores[num_elites:]

    # Adjust number of parents to select
    num_parents = min(num_select - num_elites, len(remaining_population))
    
    if num_parents > 0:
        selected_parents = roulette_wheel_selection(remaining_population, remaining_scores, num_parents)
    else:
        selected_parents = []  # No parents to select if remaining_population is empty or num_parents is 0

    # Apply crossover and mutation to selected parents
    offspring = apply_crossover(selected_parents) if selected_parents else []
    offspring = apply_mutation(offspring, mutation_rate=0.2, num_actions=env.action_space.n)

    # Combine elites and offspring to form the next generation
    next_generation = elites + offspring

    # Ensure population size consistency
    next_generation = next_generation[:num_select]

    return next_generation, max(scores) # Return next generation and best score for tracking

# Apply crossover (of selected POLICies - parents, to create new generation - offspring)
# Keep the best information about used POLICies (assessed by fitness function)
def apply_crossover(selected_policies):
    offspring = []      # Initialize list of offsprings
    for i in range(0, len(selected_policies) - 1, 2):       # Iterate over adjacent pair of selected POLICies (parents)
                                                            # even number of parents required (odd - not used)
        crossover_point = random.randint(1, len(selected_policies[0]) - 1)       # Determine random crossover point from the list of selected parents
        parent1 = selected_policies[i]                      # i selected parent
        parent2 = selected_policies[i + 1]                  # Adjacent parent (next one)
        # Create two offsprings by combining the selected parents part (determined by crossover_point)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))     # 1st part of parent1 linked with 2nd part of parent2
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        offspring.extend([child1, child2])                  # Add new children to the offspring list
    # Returns all created offsprings (new generation - new population)
    return offspring

# Apply mutation to the policies
def apply_mutation(policies, mutation_rate, num_actions):
    # POLICies - list of offsprings created by crossover
    for policy in policies:     # Iterate each POLICY
        if np.random.rand() < mutation_rate:                    # Mutation check based on random number generated (0 to 1)
            mutation_index = np.random.randint(len(policy))     # Select a random index from current POLICY (eg. [2, 1, 3, 4]) to mutate
            policy[mutation_index] = np.random.randint(num_actions)     # Assign a new random ACTION (from existing) to the exact index (ACTION) in current POLICY
    # Returns all POLICIes, some of which are mutated (altered)
    return policies

def roulette_wheel_selection(population, scores, num_select):
    min_fitness = 0.01  # Ensure no policy has zero probability
    adjusted_scores = [max(score, min_fitness) for score in scores]
    total_score = sum(adjusted_scores)

    probabilities = [score / total_score for score in adjusted_scores]

    # Select individuals based on probabilities
    selected_indices = np.random.choice(len(population), size=num_select, replace=False, p=probabilities)
    return [population[i] for i in selected_indices]

# Create a GIF
def create_gif(env, policy, title, filename, path):
    frames = []
    obs, _ = env.reset()
    done = False

    while not done:
        img = env.render()  # Render the current state
        frames.append(img)  # Add the frame to the list
        action = policy[int(obs)]  # Get action from the policy
        obs, reward, terminated, truncated, _ = env.step(action)  # Step in the environment
        done = terminated or truncated

    if reward == 1:  # Only save if the goal is reached
        frames.append(env.render())
        for i in range(len(frames)):
            img = Image.fromarray(frames[i])        # Convert frame to image
            draw = ImageDraw.Draw(img)              # Create a drawing object
            font = ImageFont.load_default()         # Load default font

            # Calculate the text position to center it
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

# Run the genetic algorithm process for a specified number of generations (evolutions)
def run_evolution(map_size, pop_size, num_generations):
    ga_directory = f'GA_solutions_slippery_True_{map_size}'
    if os.path.exists(ga_directory):
        for file_name in os.listdir(ga_directory):
            os.unlink(os.path.join(ga_directory, file_name))
    else:
        os.makedirs(ga_directory)

    env, num_actions, num_states = setup_environment(map_size)      # Set up the environment with ACTIONS and observation (STATEs)
    population = [create_random_policy(num_states, num_actions) for _ in range(pop_size)]       # Initialize population
    num_elites = 5          # Number of elite policies to preserve each generation

    # Track scores for plotting
    best_scores = []
    average_rewards = []
    average_steps = []  

    # Main evolution cycle for each generation
    for generation in range(num_generations):
        print(f"Generation: {generation + 1}")
        steps_per_policy = []  # To track the steps taken by each policy in the generation
        
        population, best_score = perform_evolution(map_size, env, population, num_elites, pop_size)  # Evolve the population

        # Calculate scores for plotting
        scores = []
        for policy in population:
            fitness = calculate_fitness(map_size, env, policy)
            scores.append(fitness)

            s, steps = run_simulation(map_size, env, policy)  # Track steps for each policy
            steps_per_policy.append(steps)

        best_scores.append(best_score)                      # Record the best score
        average_rewards.append(np.mean(scores))             # Record the average score
        average_steps.append(np.mean(steps_per_policy))     # Record the average steps taken

        # Print best and average scores for this generation + average steps taken
        print(f"Best Score: {best_score:.4f}, Average Score: {np.mean(scores):.4f}, Average Steps: {np.mean(steps_per_policy):.2f}")

    # Plotting the performance scores (separate graph)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(best_scores, label='Best Scores')
    plt.plot(average_rewards, label='Average Rewards', color='orange')
    plt.xlabel('Generations')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid()

    # Plotting the average steps
    plt.subplot(2, 1, 2)
    plt.plot(average_steps, label='Average Steps per Generation', color='green')
    plt.xlabel('Generations')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'{ga_directory}/evolution_performance.png')
    plt.show()

    # Save GIFs for successful policies
    successful_policies = []
    for i, policy in enumerate(population):
        score, s = run_simulation(map_size, env, policy)
        if score == 1:  # Check if the policy succeeded
            successful_policies.append(policy)
            create_gif(env, policy, f'Solution {i + 1}', f'solution_{i + 1}.gif', ga_directory)
    
    if not successful_policies:
        print("No successful policies found!")

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
    elif args.map_size == "8x8":
        population = 500
        generation = 300

    run_evolution(map_size = args.map_size, pop_size=population, num_generations=generation)

