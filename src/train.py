import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from collections import deque
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import os
import time
from collections import deque, namedtuple

from evaluate import evaluate_HIV, evaluate_HIV_population
import torch.nn.functional as F

class BinaryIndexedTree:
    """Class for Binary Indexed tree. This structure is used within the 
    replay buffer to store information.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(capacity + 1)  # BIT is 1-indexed
        self.data = np.zeros(capacity, dtype=object)
        self.position = 0
        self.size = 0

    def add(self, idx, value):
        while idx <= self.capacity:
            self.tree[idx] += value
            idx += idx & -idx

    def sum(self, idx):
        total = 0
        while idx > 0:
            total += self.tree[idx]
            idx -= idx & -idx
        return total

    def update(self, idx, new_value):
        current_value = self.sum(idx) - self.sum(idx - 1)
        self.add(idx, new_value - current_value)

    def prefix_sum(self, value):
        idx = 0
        bit_mask = self.capacity.bit_length()
        for i in reversed(range(bit_mask)):
            next_idx = idx + (1 << i)
            if next_idx <= self.capacity and value > self.tree[next_idx]:
                idx = next_idx
                value -= self.tree[next_idx]
        return idx + 1  # Convert to 1-indexed

    def add_data(self, priority, data):
        idx = self.position + 1  # 1-indexed for BIT
        self.data[self.position] = data
        self.update(idx, priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class PrioritizedReplayBuffer:
    """The Prioritized Replay Buffer is a data structure to store information
    during the training process. It uses binary indexed trees as an internal
    structure. 
    """
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=1e-4):
        self.tree = BinaryIndexedTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add_data(priority, data)

    def sample(self, batch_size):
        batch_indices = []
        batch = []
        segment = self.tree.sum(self.tree.size) / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            idx = self.tree.prefix_sum(value)  # Get index using prefix sum
            data_idx = (idx - 1) % self.capacity  # Convert to 0-indexed
            batch_indices.append(idx)
            priorities.append(self.tree.sum(idx) - self.tree.sum(idx - 1))
            batch.append(self.tree.data[data_idx])

        sampling_probabilities = np.array(priorities) / self.tree.sum(self.tree.size)
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, batch_indices, is_weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = float(priority)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)

    def __len__(self):
        return self.tree.size


class DuelingQNetwork(nn.Module):
    """The Dueling Q Network is a neural network that embeds values, 
    features and advantages. From these, we calculate q values during training.  
    """
    def __init__(self, state_dim=6, action_dim=4):
       super(DuelingQNetwork, self).__init__()

       self.feature = nn.Sequential(
           nn.Linear(state_dim, 256),
           nn.SiLU(),
           nn.Linear(256, 512),
           nn.SiLU(),
           nn.Linear(512, 1024),
           nn.SiLU(),
           nn.Linear(1024, 1024),
           nn.SiLU(),
           nn.Linear(1024, 1024),
           nn.SiLU(),
       )
       self.value_stream = nn.Sequential(
           nn.Linear(1024, 512),
           nn.SiLU(),
           nn.Linear(512, 256),
           nn.SiLU(),
           nn.Linear(256, 1)
       )
       self.advantage_stream = nn.Sequential(
           nn.Linear(1024, 512),
           nn.SiLU(),
           nn.Linear(512, 256),
           nn.SiLU(),
           nn.Linear(256, action_dim)
       )

       self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
       features = self.feature(x)
       value = self.value_stream(features)              
       advantage = self.advantage_stream(features)      
       q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
       return q_values



class ProjectAgent:
    """The ProjectAgent class unites the previous classes together into one
    functioning structure. In the initialization, we set hyperparameters used during training.
    """
    def __init__(self, state_dim=6, n_actions=4):
        self.n_actions = n_actions # Number of actions
        self.state_dim = state_dim # Number of states
        self.gamma = 0.85 # Gamma value for discounted reward 
        self.save_path = "project_agent_trained_for_longer.pt" # Save path
        
        self.epsilon = 1.0 # Epsilon start value
        self.epsilon_min = 0.01 # Minimum Epsilon value
        self.epsilon_decay = 0.9965 # Epsilon decrease
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device to train on
        
        self.q_network = DuelingQNetwork(state_dim, n_actions).to(self.device) # Q network init
        self.target_network = DuelingQNetwork(state_dim, n_actions).to(self.device) # Taget network init
        self.target_network.load_state_dict(self.q_network.state_dict()) 
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3, betas=(0.5, 0.999)) # Setting up optimiizer
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=350, gamma=0.5) # Learning rate scheduler
        
        self.replay_buffer = PrioritizedReplayBuffer(capacity=600000, alpha=0.6, beta=0.4) # Replay buffer with binary trees
        
        self.batch_size = 1024
        self.target_update_freq = 1000 # Frequency to update  
        self.update_count = 0

    def act(self, observation, use_random=False):
        """Function to select an action. This can be done either randomly, or based on the trained q network.
        Parameters:
        observation -> observation seen during training
        use_random  -> boolean value whether to use random selection of action or not
        """
        if use_random and random.random() < self.epsilon:
           return np.random.randint(0, self.n_actions)
       state_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
       q_values = self.q_network(state_t)
       return q_values.argmax(dim=1).item()

    def train_step(self):
        """Function to execute a train step. The states, actions, rewards, etc are firstly sampled from the buffer and
        converted into tensors. Then, the Q Network is used to calculate the current q values. Next actions, 
        q values and target q values are then calculated and the difference between the target q values and the current 
        q values obtained. From this, the loss is calculated which is used to update the q networks.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        # Sample from buffer     
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        # Convert everything into tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        # Calculate the current q values
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        # Obtain next q values and target values 
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(1)
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        # Calculate errors between current q values and target q values
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        # Update buffer priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        # Calculate the loss
        loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()
        # Update model and optimizer, clip gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # If necessary, update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def step_scheduler(self):
        self.scheduler.step()
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="./src/project_agent_trained_for_longer.pt"):
        #path = "./project_agent_trained_for_longer.pt"
        print(f"loading from {path}")
        weights = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(weights)
        self.q_network.eval()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

def train_agent(env_name="HIVPatient-v0", num_episodes=1000, max_steps_per_episode=200, save_freq=50):
    """Function to navigate training the agent. At first, the environment is loaded and the ProjectAgent 
    initialized. Then, we iterate over episodes, select actions and update our parameters. After each
    action selection, we update the buffer with the previously selected action, reward and next state. 
    Then, the model parameters are updated and we transition into the next state.
    """
    # Initialize the environment
    env = TimeLimit(HIVPatient(), max_episode_steps=max_steps_per_episode)
    agent = ProjectAgent()

    rewards_per_episode = []
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            # Select action by agent and use random if value smaller than epsilon
            action = agent.act(state, use_random=True)

            # Take action in environment
            next_state, reward, done, _, _ = env.step(action)
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            total_reward += reward

            # Update agent
            agent.train_step()

            # Move to the next state
            state = next_state

            if done:
                break

        # Update epsilon and learning rate scheduler
        agent.update_epsilon()
        agent.step_scheduler()

        rewards_per_episode.append(total_reward)

        # Log progress on average  of 10 episodes
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_per_episode[-10:])
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f} | Time: {elapsed_time:.2f}s")

        # Save model every save_freq steps
        if episode % save_freq == 0:
            agent.save(agent.save_path)

    # Final save of the model
    agent.save(agent.save_path)
    print("Training completed and model saved.")

    return rewards_per_episode

if __name__ == "__main__":
    env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
    rewards = train_agent(env_name="HIVPatient-v0", num_episodes=5000, max_steps_per_episode=200, save_freq=50)


    
