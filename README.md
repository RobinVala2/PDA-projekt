# PDA - project: Frozen Lake

## Team members:
| Name                 | Id      | Username   |
|----------------------|---------|------------|
| Bc. Jaroslav Spolek  | 241105  | Jarin02    |
| Bc. Lada Struziakova | 241110  | Sailynka   |
| Bc. Samuel Sulka     | 241116  | ssulka     |
| Bc. Robin Vala       | 241124  | RobinVala2 |

## Reinforcment Learning

### Features
1. Train a Q-learning agent on 4x4 or 8x8 Frozen Lake maps.
2. Option to enable or disable slipperiness in the environment.
3. Save and load trained Q-table for evaluation.
4. Visualize training performance with reward, step, and success rate plots.
5. Generate GIFs of the agent's best solutions during evaluation.

### How It Works
**Training**: The agent uses Q-learning to update its Q-table based on rewards and transitions. It balances exploration and exploitation with an epsilon-greedy policy, which decays over episodes.

**Evaluation**: The agent uses a pre-trained Q-table to execute actions, maximizing the Q-values for the current state. The success rate and GIFs of the best solutions are saved during this phase.

### Usage
Run the script with following arguments:
| Argument             | Description          | Default Value        |
|----------------------|----------------------|----------------------|
| --episodes  | Number of episodes for training or testing  | 10000    |
| --training | Set to True for training, False for testing  | True   |
| --map_name     | Size of the map: 4x4 or 8x8  | 8x8     |
| --is_slippery      | Enable or disable slippery tiles in the environment  | True |

**Example Usage:**

Train the agent on an 8x8 slippery map:
```sh
python q_table_run.py --episodes 10000 --training True --map_name "8x8" --is_slippery True
```
Train the agent on a 4x4 non-slippery map:
```sh
python q_table_run.py --episodes 15000 --training True --map_name "4x4" --is_slippery False
```
Evaluate the agent's performance using a pre-trained Q-table on an 8x8 slippery map:
```sh
python q_table_run.py --episodes 1000 --training False --map_name "8x8" --is_slippery True
```
Evaluate on a 4x4 non-slippery map:
```sh
python q_table_run.py --episodes 1000 --training False --map_name "4x4" --is_slippery False
```

### Outputs
1. Q-table:
    - Saved as frozen_lake{map_name}.pkl in Q-table_solutions_{map_name}_{is_slippery}.
    - Final Q-table saved as last_q_table.txt.
2. Training Plots:
    - Rewards per episode.
    - Steps per episode.
    - Cumulative success rate.
    - Saved as frozen_lake_episode{map_name}.png.
3. GIFs:
    - GIFs of the best solutions (minimum steps) are saved during evaluation.

<i>After training and evaluation, the success rate and steps per episode provide insights into the agent's performance. Success rates can vary depending on map size, slipperiness, and the number of episodes.</i> 

### Functions 
```initialize_q_table(state_space, action_space)```
    
- **Purpose**: Creates an initial Q-table filled with zeros.
- **Inputs**:
    - ```state_space```: The number of states in the environment.
    - ```action_space```: The number of possible actions in the environment.
- **Outputs**:
    - A Q-table (```numpy``` array) of shape ```(state_space, action_space)``` initialized to zeros.

<hr>

```load_q_table(map_name, path)```
- **Purpose**: Loads a pre-trained Q-table from a file.
- **Inputs**:
    - ```map_name```: The name of the map for which the Q-table was trained (e.g., 4x4, 8x8).
    - ```path```: Path to the folder containing the saved Q-table.
- **Outputs**:
    - The loaded Q-table as a ```numpy``` array.

<hr>

```choose_action(epsilon, action_space, q_table, state)```
- **Purpose**: Determines the action to take based on an epsilon-greedy policy.
    - **Exploration**: The agent chooses a random action to discover new strategies and learn about the environment.
    - **Exploitation**: The agent selects the action that maximizes the known Q-value for the current state, based on what it has learned so far.
- **Inputs**:
    - ```epsilon```: Probability of choosing a random action (exploration).
    - ```action_space```: Action space object to sample random actions from.
    - ```q_table```: The current Q-table.
    - ```state```: The agent's current state.
- **Outputs**:
    - An action (integer) based on either exploration or exploitation.

<hr>

```run_agent(episodes, training, map_name, is_slippery)```
- **Purpose**: The core method that trains or evaluates the Q-learning agent.
- **Inputs**:
    - ```episodes```: Number of episodes to run.
    - ```training```: Boolean flag to determine if the agent is in training mode.
    - ```map_name```: Name of the map (4x4 or 8x8).
    - ```is_slippery```: Boolean flag to enable or disable slipperiness.
- **Key Features**:
    - Training Phase:
        - Initializes the Q-table.
        - Updates the Q-table using the Bellman equation.
        - Tracks metrics: rewards, steps, and cumulative success.
        - Saves the trained Q-table and performance plots.
    - Evaluation Phase:
        - Loads the pre-trained Q-table.
        - Evaluates the agent’s performance over multiple episodes.
        - Measures success rate and saves GIFs for the best solutions.

### Hyperparameters Description

| Parameter            | Value    | Description                                                                                 |
|----------------------|----------|---------------------------------------------------------------------------------------------|
| `learning_rate`      | `0.8`    | Determines how much new information overrides old information during Q-value updates.       |
| `discount_factor`    | `0.95`   | Represents the importance of future rewards compared to immediate rewards.   |
| `epsilon`            | `1`      | Initial exploration probability; controls the balance between exploration and exploitation. |
| `min_epsilon`        | `0.01`   | Minimum value of `epsilon` to ensure some exploration continues during training.            |
| `epsilon_decay`      | `0.0001` | Linear decay factor for reducing `epsilon` after each episode.                              |
| `max_steps_per_episode` | `200` | Maximum number of steps the agent can take per episode to prevent infinite loops.           |



### Epsilon Greedy Strategy
The epsilon-greedy strategy is a simple method used in Reinforcement Learning to balance two conflicting objectives:
1. **Exploration**: Discovering new actions and states to learn about the environment.
    - The goal is to gather knowledge that might lead to better policies in the long run.
    - Example: Trying actions that weren’t used much before to see if they might lead to higher rewards.
2. **Exploitation**: Leveraging the knowledge the agent has already acquired to maximize immediate rewards.
    - The goal is to act optimally based on the current learned policy or knowledge.
    - Example: Choosing the best-known action (so far) for the current state to gain a higher reward.

**How It Works**

1. **Epsilon (ϵ)**:
    - ϵ is a parameter that controls the probability of exploration.
    - Value ranges between 0 and 1:
        - ϵ=1: Full exploration (completely random actions).
        - ϵ=0: Full exploitation (always chooses the best-known action).
    - Over time, ϵ is often decayed to shift the balance from exploration to exploitation as the agent learns more.

2. **Decision Process**:
    - At each time step or state:
        - Generate a random number r between 0 and 1.
        - Compare r to ϵ:
            - If r<ϵ: Explore → Take a random action.
            - If r≥ϵ: Exploit → Take the action with the highest Q-value for the current state.

### Q-Value Update Equation

The Q-value update equation is based on the Bellman Equation, which helps the agent balance immediate rewards and future rewards by iteratively refining its estimates of the optimal policy.

```python
q_table[state][action] = q_table[state][action] + learning_rate * (
    reward + discount_factor * np.max(q_table[new_state]) - q_table[state][action]
)
```

| **Component**                                    | **Explanation**                                                                                                     |
|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `q_table[state][action]`                         | The current Q-value for the specific state-action pair. This represents the agent's estimate of the action's value. |
| `learning_rate`                 | Determines how much the new information (reward and future values) influences the updated Q-value.                 |
| `reward`                              | The immediate reward received after performing the action in the current state.                                     |
| `discount_factor`               | Controls the importance of future rewards. A higher value prioritizes long-term rewards over short-term ones.       |
| `np.max(q_table[new_state])`                     | The maximum Q-value of the next state (`new_state`). It represents the best possible action-value for the next step.|
| `reward + discount_factor * np.max(q_table[new_state])` | The **target Q-value**—an estimate of the best achievable total reward starting from the current state-action pair.  |
| `reward + discount_factor * np.max(...) - q_table[state][action]` | The **Temporal Difference (TD) Error**—measures how far the current Q-value is from the target Q-value.             |
| `learning_rate * (...)`                          | Scales the impact of the TD error on the Q-value update. Ensures stable learning by applying updates incrementally.  |


## Genetic Algorithm (Slippery True)

### Features
1. Customizable Environment:
    - Supports 4x4 and 8x8 FrozenLake maps.
    - Designed specifically for environments with slippery=True to add complexity.
2. Genetic Algorithm Implementation:
    - Population-based Search: Each individual in the population represents a policy.
    - Evolutionary Operators: Includes selection, crossover, and mutation.
3. Visualization:
    - Plots best scores, average rewards, and average steps per generation.
    - Creates GIFs of successful policies in action.

### Genetic Algorithm Process
1. Initialize Population:
    - Each individual (policy) is randomly initialized.
    - A policy maps states to actions.
2. Selection:
    - Individuals are evaluated using a fitness function, which measures their ability to reach the goal efficiently.
    - The top-performing individuals (elites) are preserved.
3. Crossover:
    - Pairs of parents are combined to produce offspring, sharing strategies between policies.
4. Mutation:
    - Random changes are applied to some policies to introduce diversity.
5. Repeat:
    - The process continues for a specified number of generations until a termination condition is met.
6. Save Results:
    - Successful policies are saved as GIFs.
    - Plots showing performance metrics are generated.

### Usage
The script can be run with the following arguments:
```sh
python ga_run-slipperyTrue.py --map_size <MAP_SIZE>
```

| Argument                 | Description      | Default Value   | Options|
|----------------------|---------|------------|----------|
| `--map_size`  | Size of the FrozenLake map  | "4x4"    |4x4, 8x8|

### Key Parameters

| Parameter                 | Description      | 
|----------------------|---------|
| `population`  | Number of policies in each generation.  | 
| `num_generations`  | Number of generations for the evolution process.  | 
| `num_elites`  | Number of top-performing policies preserved in each generation.  | 
| `mutation_rate`  | Probability of mutating a policy.  | 

**For the 4x4 map**:
   - population = 200
   - num_generations = 200

**For the 8x8 map**:
   - population = 500 
   - num_generations = 300

### Outputs
1. Performance Plots:
    - Saved as evolution_performance.png in the results folder.
    - Includes:
        - Best scores per generation.
        - Average rewards.
        - Average steps.
2. GIFs:
    -  GIFs of successful policies in action are saved in the results folder.
3. Console Logs:
    - Details of each generation, including:
        - Best score.
        - Average reward.
        - Average steps.

### Important Notes
**Slippery Environment**: This script is specifically designed for slippery=True, where movements are less deterministic, making the problem more challenging.

***The algorithm is optimized to handle the unpredictability of slippery tiles by leveraging genetic evolution to discover robust policies.***

### Functions 

```create_random_policy(num_states, num_actions)```

- **Purpose**: Generates a random policy (strategy) mapping states to actions.
- **Inputs**:
    - `num_states`: Total number of states in the environment.
    - `num_actions`: Total number of possible actions.
- **Outputs**:
    - A numpy array where each element corresponds to a randomly selected action for each state.

<hr>

```run_simulation(map_size, env, policy)```
- **Purpose**: Runs a single game simulation in the environment using the provided policy.
- **Inputs**:
    - `map_size`: Size of the FrozenLake map (4x4 or 8x8).
    - `env`: The FrozenLake environment.
    - `policy`: A policy (array mapping states to actions).
- **Outputs**:
    - (1, steps) if the agent successfully reaches the goal.
    - (0, steps) if the agent falls into a hole.

<hr>

```calculate_fitness(map_size, env, policy, num_simulations=10)```
- **Purpose**: Evaluates a policy's effectiveness by simulating multiple episodes.
- **Inputs**:
    - `map_size`: Size of the map.
    - `env`: The FrozenLake environment.
    - `policy`: The policy to be evaluated.
    - `num_simulations`: Number of simulations to run (default: 10).
- **Outputs**:
    - `raw_fitness`: A fitness score based on the average success rate and efficiency (steps taken). Higher values indicate better policies.

<hr>

```perform_evolution(map_size, env, population, num_elites, num_select)```
- **Purpose**: Evolves the current population using selection, crossover, and mutation.
- **Inputs**:
    - `map_size`: Size of the FrozenLake map.
    - `env`: The FrozenLake environment.
    - `population`: List of policies (current generation).
    - `num_elites`: Number of top-performing policies to preserve directly.
    - `num_select`: Total number of policies to select for the next generation.
- **Outputs**:
    - `next_generation`: The evolved population (next generation).
    - `best_score`: The highest fitness score of the current generation.

<hr>

```apply_crossover(selected_policies)```
- **Purpose**: Combines pairs of selected policies to create offspring.
- **Inputs**:
    - `selected_policies`: Policies chosen as parents for crossover.
- **Outputs**:
    - `offspring`: A list of new policies (children) generated from the parent policies.

<hr>

```apply_mutation(policies, mutation_rate, num_actions)```
- **Purpose**: Introduces randomness to the population by mutating some policies.
- **Inputs**:
    - `policies`: A list of policies to mutate.
    - `mutation_rate`: Probability of mutating each policy.
    - `num_actions`: Total number of possible actions in the environment.
- **Outputs**:
    - The mutated policies, with some random changes applied.

<hr>

```roulette_wheel_selection(population, scores, num_select)```
- **Purpose**: Selects policies based on their fitness scores using a probabilistic approach.
- **Inputs**:
    - `population`: List of current policies.
    - `scores`: Fitness scores for each policy.
    - `num_select`: Number of policies to select.
- **Outputs**:
    - A list of selected policies for the next generation.


## Genetic Algorithm (Slippery False)
### Features
1. Customizable Environment:
    - Supports 4x4 and 8x8 FrozenLake maps.
    - Specifically designed for environments with slippery=False, where movements are deterministic.
2. Genetic Algorithm Implementation:
    - Parallelized Fitness Evaluation: Speeds up the evaluation process by using multiple CPU cores.
    - Evolutionary Operators: Includes fitness proportional selection, crossover, and mutation.
3. Visualization:
    - Plots best scores, average rewards, and average steps per generation.
    - Creates GIFs of successful policies in action.

### Genetic Algorithm Process
1. Initialize Population:
    - Each individual (policy) is randomly initialized.
    - A policy maps states to actions.
2. Fitness Evaluation:
    - Fitness is calculated based on:
        - Reaching the goal (maximum reward for success).
        - Visiting new states (exploration reward).
        - Minimizing the number of steps (penalizing excessive steps).
    - Parallel processing is used for faster evaluation.
3. Selection:
    - Policies are selected based on fitness using fitness proportional selection.
4. Crossover:
    - Combines pairs of selected policies to produce offspring, sharing strategies between parents.
5. Mutation:
    - Introduces random changes to some offspring policies to encourage exploration.
6. Repeat:
    - The process continues for a specified number of generations.
7. Save Results:
    - Successful policies are saved as GIFs.
    - Plots showing performance metrics are generated.

### Usage
The script can be run with the following arguments:
```sh
python ga_run-slipperyFalse.py --map_size <MAP_SIZE>
```

| Argument                 | Description      | Default Value   | Options|
|----------------------|---------|------------|----------|
| `--map_size`  | Size of the FrozenLake map  | "4x4"    |4x4, 8x8|

### Key Parameters

| Parameter                 | Description      | 
|----------------------|---------|
| `population`  | Number of policies in each generation.  | 
| `num_generations`  | Number of generations for the evolution process.  | 
| `mutation_rate`  | Probability of mutating a policy.  | 

**For the 4x4 map**:
   - population = 200
   - num_generations = 200
   - max_steps = 50

**For the 8x8 map**:
   - population = 500 
   - num_generations = 300
   - max_steps = 100

### Outputs
1. Performance Plots:
    - Saved as evolution_performance.png in the results folder.
    - Includes:
        - Best scores per generation.
        - Average rewards.
        - Average steps.
2. GIFs:
    -  GIFs of successful policies in action are saved in the results folder.
3. Console Logs:
    - Details of each generation, including:
        - Best score.
        - Average reward.
        - Average steps.

### Important Notes
**Deterministic Environment**: This script is specifically designed for slippery=False, where movements are deterministic and predictable.

***The algorithm leverages parallel processing to evaluate policies efficiently.***

### Functions 

```create_random_policy(num_states, num_actions)```

- **Purpose**: Generates a random policy (strategy) mapping states to actions.
- **Inputs**:
    - `num_states`: Total number of states in the environment.
    - `num_actions`: Total number of possible actions.
- **Outputs**:
    - A numpy array where each element corresponds to a randomly selected action for each state.

<hr>

```run_simulation(env, policy, max_steps)```
- **Purpose**: Runs a single game simulation in the environment using the provided policy.
- **Inputs**:
    - `env`: The FrozenLake environment.
    - `policy`: A policy (array mapping states to actions).
    - `max_steps`: Maximum steps allowed per episode to prevent infinite loops..
- **Outputs**:
    - `total_reward`: The cumulative reward for the episode.
    - `steps`: The total number of steps taken.

<hr>

```calculate_fitness_parallel(population, map_name, is_slippery, max_steps)```
- **Purpose**: Calculates the fitness of each policy in the population in parallel for faster execution.
- **Inputs**:
    - `population`: List of policies to evaluate.
    - `map_name`: Size of the map.
    - `is_slippery`: Boolean indicating slipperiness.
    - `max_steps`: Maximum steps allowed per episode.
- **Outputs**:
    - A list of fitness scores for the population.

<hr>

```fitness_proportional_selection(population, scores, num_select)```
- **Purpose**: Selects policies for reproduction based on their fitness scores.
- **Inputs**:
    - `population`: List of policies (current generation).
    - `scores`: Fitness scores corresponding to the population.
    - `num_select`: Number of policies to select.
- **Outputs**:
    - A list of selected policies for reproduction.

<hr>

```apply_crossover(selected_policies)```
- **Purpose**: Combines pairs of selected policies to create offspring.
- **Inputs**:
    - `selected_policies`: Policies chosen as parents for crossover.
- **Outputs**:
    - `offspring`: A list of new policies (children) generated from the parent policies.

<hr>

```apply_mutation(policies, mutation_rate, num_actions)```
- **Purpose**: Introduces randomness to the population by mutating some policies.
- **Inputs**:
    - `policies`: A list of policies to mutate.
    - `mutation_rate`: Probability of mutating each policy.
    - `num_actions`: Total number of possible actions in the environment.
- **Outputs**:
    - The mutated policies, with some random changes applied.

### Proportional Selection Methods 

The project includes two proportional selection methods to choose individuals for reproduction based on their fitness scores:

1. **Fitness Proportional Selection** used in `ga_run-slipperyFalse`
1. **Roulette Wheel Selection** used in `ga_run-slipperyTrue`

**Comparison**:

| Feature                        | **Fitness Proportional Selection**                      | **Roulette Wheel Selection**                      |
|--------------------------------|---------------------------------------------------------|--------------------------------------------------|
| **Non-Negative Scores**         | Adjusts negative scores explicitly.                    | Uses a minimum fitness threshold (`min_fitness`). |
| **Zero Total Fitness Handling** | Assigns equal probabilities to all individuals.        | Relies on adjusted fitness scores.               |
| **Edge Case Handling**          | Handles insufficient probabilities with uniform fallback. | Assumes fitness scores are well-preprocessed.    |
| **Complexity**                  | More robust, with additional checks for edge cases.    | Simpler and requires fewer checks.               |


## Differences between GA scripts

| **Aspect**                          | **File 1** (Slippery `True`)                                                                                              | **File 2** (Slippery `False`)                                                                                             |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **Environment Slipperiness**        | `is_slippery=True`: Adds randomness to movements, making the problem harder to solve.                                     | `is_slippery=False`: Movements are deterministic, making the environment more predictable.                                |
| **Selection Method**                | Includes **`roulette_wheel_selection`**: Uses a simpler fitness-proportional selection approach with fewer edge-case checks. | Uses **`fitness_proportional_selection`**: Handles negative scores and edge cases explicitly, making it more robust.       |
| **Parallel Fitness Evaluation**     | Not implemented.                                                                                                          | Implements **parallelized fitness evaluation** using Python’s `multiprocessing.Pool`, speeding up policy evaluations.      |
| **Exploration Reward in Fitness**   | No explicit exploration reward mechanism.                                                                                 | Introduces an **exploration reward** (`+0.1` for visiting new states) in fitness evaluation to encourage diverse policies.  |
| **Step Penalty in Fitness**         | Does not penalize steps explicitly.                                                                                       | Includes a **step penalty** (`-0.01` per step) in fitness evaluation to prioritize efficiency.                             |
| **Population Diversification**      | Relies on mutation for introducing diversity.                                                                              | Adds random policies to the population every generation (`pop_size // 5`) to ensure exploration of new strategies.         |
| **Fitness Score Averaging**         | Evaluates fitness over a fixed number of episodes, but fitness calculation is simpler.                                     | Averages fitness scores over multiple episodes (`10` by default), providing a more stable measure of policy effectiveness. |
| **Generation Structure**            | Simpler implementation of crossover and mutation.                                                                         | Introduces **elite preservation** (top policies) to ensure the best-performing policies are retained across generations.   |
| **Policy Evaluation**               | Policies are evaluated episode-by-episode sequentially.                                                                   | Policies are evaluated in **parallel**, significantly improving performance for large populations.                         |
