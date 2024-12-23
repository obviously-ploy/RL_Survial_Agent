from environment.grid_world import GridWorld
from config import ENV_CONFIG as config
from agents.random_agent import random_agent
from agents.q_learning import test_q_learning_with_params
from agents.ppo import test_PPO_with_params
from agents.A2C import test_a2c_testing_with_params
import numpy as np
import matplotlib.pyplot as plt

def plot_rewards_comparison(q_rewards, a2c_rewards, ppo_rewards, dqn_rewards, random_rewards, title):
    plt.figure()

    
    timesteps = np.arange(1, len(a2c_rewards) + 1) 
    plt.plot(timesteps, q_rewards, label="Q-Learning", color="blue", linestyle="-")
    plt.plot(timesteps, a2c_rewards, label="A2C", color="green", linestyle="-")
    plt.plot(timesteps, ppo_rewards, label="PPO", color="orange", linestyle="-")
    plt.plot(timesteps, random_rewards, label="Random", color="red", linestyle=":")
    
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig(f'{title}_graph.png')

def plot_rewards_comparison_without_random(q_rewards, a2c_rewards, ppo_rewards, title):
    plt.figure()

    
    timesteps = np.arange(1, len(a2c_rewards) + 1)  # X-axis values (1-indexed)
    plt.plot(timesteps, q_rewards, label="Q-Learning", color="blue", linestyle="-")
    plt.plot(timesteps, a2c_rewards, label="A2C", color="green", linestyle="-")
    plt.plot(timesteps, ppo_rewards, label="PPO", color="orange", linestyle="-")
    
    
    plt.title(title)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig(f'{title}_graph.png')


def run_experiment(map_name, params_grid):
    print(f"Running experiments on {map_name}")
    grid_world = GridWorld(config, map_name)

    # Test PPO
    PPO_results = test_PPO_with_params(grid_world, params_grid)
    best_PPO = max(PPO_results, key=lambda x: x[-1])
    print(f"Best PPO Params for {map_name}: {best_PPO[:2]}")
    
    # Test Q-Learning
    
    q_learning_results = test_q_learning_with_params(grid_world, params_grid)
    best_q_learning = max(q_learning_results, key=lambda x: x[-1])  
    print(f"Best Q-Learning Params for {map_name}: {best_q_learning[:3]}")

    # Test A2C
    a2c_results = test_a2c_testing_with_params(grid_world, params_grid)
    best_a2c = max(a2c_results, key=lambda x: x[-1])  
    print(f"Best A2C Params for {map_name}: {best_a2c[:3]}")


    random_rewards = random_agent(grid_world)
    
    plot_rewards_comparison(
        best_q_learning[3], best_a2c[3], best_PPO[2], [], random_rewards,
        f"Algorithm Comparison ({map_name})"
    )
    
    plot_rewards_comparison_without_random(
        best_q_learning[3], best_a2c[3], best_PPO[2],
        f"Algorithm Comparison Without Baseline ({map_name})"
    )

def main():
    map_names = ['all-land-map', 'river-map', 'lakes-map', 'island-map']

    params_grid = {
        "gamma": [0.8, 0.9, 0.95, 0.99],
        "step_size": [0.001, 0.005, 0.01, 0.1],
        "epsilon": [0.01, 0.1, 0.25, 0.5]
    }

   
    for map_name in map_names:
        run_experiment(map_name, params_grid)

      


        
        
    
    





if __name__ == "__main__":
    main()
