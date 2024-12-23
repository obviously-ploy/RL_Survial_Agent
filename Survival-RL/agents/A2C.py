import numpy as np
class ActorCriticAgent:
    def __init__(self, env, gamma=0.5, step_size=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma 
        self.step_size = step_size  

        self.state_dim = np.prod(env.grid_size) * 8  
        self.action_dim = env.n_actions  
        self.actor_weights = np.random.rand(self.state_dim, self.action_dim)  
        self.critic_weights = np.random.rand(self.state_dim)  

    def feature_vector(self, state):
        """Generate a feature vector for the given state."""
        feature = np.zeros(self.state_dim)
        index = (state[0] * self.env.grid_size[1] + state[1]) * 8 + state[2]  
        feature[index] = 1 
        return feature

    def policy(self, state):
        """Compute the policy distribution for the given state."""
        feature = self.feature_vector(state)
        preferences = np.dot(self.actor_weights.T, feature)
        exp_preferences = np.exp(preferences - np.max(preferences)) 
        return exp_preferences / np.sum(exp_preferences)

    def choose_action(self, state):
        """Select an action based on the policy distribution."""
        probabilities = self.policy(state)
        return np.random.choice(self.action_dim, p=probabilities)

    def value(self, state):
        """Compute the value of the given state."""
        feature = self.feature_vector(state)
        return np.dot(self.critic_weights, feature)

    def learn(self, state, action, reward, next_state, terminated):
        """Update the actor and critic weights using TD error."""
        feature = self.feature_vector(state)
        next_value = 0 if terminated else self.value(next_state)

        td_target = reward + self.gamma * next_value
        td_error = td_target - self.value(state)

        self.critic_weights += self.step_size * td_error * feature

        policy_dist = self.policy(state)
        grad_log_pi = feature[:, None] * (np.eye(self.action_dim)[action] - policy_dist)
        self.actor_weights += self.step_size * td_error * grad_log_pi

    def train(self, max_episodes=1000, evaluate_every=20):
        total_rewards = []

        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset()
            terminated = False
            episode_reward = 0

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)

                self.learn(state, action, reward, next_state, terminated)

                state = next_state
                episode_reward += reward

            if episode % evaluate_every == 0:
                total_rewards.append(episode_reward)

        return total_rewards, self.actor_weights, self.critic_weights

    def evaluate(self, episodes=500):
        total_rewards = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            terminated = False
            episode_reward = 0

            while not terminated:
                probabilities = self.policy(state)
                action = np.argmax(probabilities)  
                next_state, reward, terminated, _, _ = self.env.step(action)

                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards


def test_a2c_testing_with_params(env, param_grid):
    results = []  # Store the results as a list of tuples
    
    for gamma in param_grid['gamma']:
        for step_size in param_grid['step_size']:
            for epsilon in param_grid['epsilon']:
                print(f"Testing A2C with gamma={gamma}, step_size={step_size}, epsilon={epsilon}...")
                
                agent = ActorCriticAgent(env, gamma=gamma, step_size=step_size, epsilon=epsilon)
                
                # Train the agent
                rewards, actor_weights, critic_weights = agent.train()
                
                
                avg_reward = np.mean(rewards)
                # Store results
                results.append((gamma, step_size, epsilon, rewards, avg_reward))
            
    
    return results
