import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


# Network for the model
class PPONetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), # choose 128 as number of features
            nn.ReLU()
        )
        # outputs probabilities of all actions
        self.actor = nn.Sequential(
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        # changes state value into scalar
        self.critic = nn.Linear(128, 1)

    # takes the state and processes it in the network
    def forward(self, state):
        shared_out = self.shared(state)
        action_probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_probs, value
    
def PPO(env, input_dim=5, n_actions=9, gamma=0.99, clip_epsilon=0.2, lr=1e-4, k_epochs=4, max_episodes=1000, evaluate_every=20):
    model = PPONetwork(input_dim, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def compute_returns_and_advantages(rewards, values, dones):
        returns = []
        advantages = []
        G = 0
        A = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                G = 0
                A = 0
            G = rewards[i] + gamma * G
            td_error = rewards[i] + gamma * (0 if dones[i] else values[i + 1]) - values[i]
            A = td_error + gamma * A
            returns.insert(0, G)
            advantages.insert(0, A)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        return returns, advantages
    
    tot_reward = []
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        state = preprocess_state(state)

        states = []
        actions = []
        rewards = []
        dones = []
        old_log_probs = []
        values = []

        terminated = False
        while not terminated:
            # get action probabilities and value estimate
            action_probs, value = model(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            next_state, reward, terminated, _, _ = env.step(action.item())
            next_state = preprocess_state(next_state)

            # store trajectory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated)
            old_log_probs.append(action_dist.log_prob(action).detach())
            values.append(value.detach().item())

            state = next_state

        values.append(0)  # random value for terminal state
        returns, advantages = compute_returns_and_advantages(rewards, values, dones)

        # change trajectory to tensor
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.stack(old_log_probs)

        # optimize policy and value network by going through
        # the network k_epoch times
        for _ in range(k_epochs):
            action_probs, new_values = model(states)
            new_action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = new_action_dist.log_prob(actions)

            # ratio for policy
            ratios = torch.exp(new_log_probs - old_log_probs)

            # surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # MSE for critic loss
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)

            # total loss
            loss = actor_loss + 0.5 * critic_loss

            # backpropagate 
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        curr_reward = sum(rewards)
        if episode % evaluate_every == 0:
            tot_reward.append(curr_reward)
    return tot_reward

def preprocess_state(state):
    # state = (x, y, wood, boat, sword)
    return torch.tensor(state, dtype=torch.float32)


def test_PPO_with_params(grid_world, param_grid):
    results = []
    for gamma in param_grid['gamma']:
        for epsilon in param_grid['epsilon']:
            print(f"Testing PPO with gamma={gamma} and epsilon={epsilon}...")
            rewards = PPO(
                env=grid_world,
                gamma=gamma,
                clip_epsilon=epsilon,
            )
            avg_reward = np.mean(rewards)
            results.append((gamma,epsilon, rewards, avg_reward))
    return results

