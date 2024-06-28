import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=100000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        action_values = self.model(state)
        return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples in the memory yet

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward  # Start with the immediate reward
            if not done:  # If next state is not terminal
                # Predict future discounted reward
                target += self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state).unsqueeze(0))).item()

            # Get the predicted Q-values from the current state
            current_q_values = self.model(torch.FloatTensor(state).unsqueeze(0))
            # Update the Q-value for the action taken
            current_q_values[0][action] = target

            # Perform a gradient descent step on (current_q_values - target)^2
            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q_values, self.model(torch.FloatTensor(state).unsqueeze(0)))
            loss.backward()
            self.optimizer.step()

        # Reduce exploration rate epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay