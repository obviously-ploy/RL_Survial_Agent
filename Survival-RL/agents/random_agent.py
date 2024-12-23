import numpy as np

def random_agent(env, num_episodes=50):
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()  
        episode_reward = 0
        terminated = False

        while not terminated:
            action = np.random.choice(env.n_actions)  
            state, reward, terminated, _, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    return total_rewards