import gym
import numpy as np
from gym import spaces
import traci
from sumolib import checkBinary
from enhance_envi import SumoEnv
from envy_static import SumoEnv2
import torch
import matplotlib.pyplot as plt

def evaluate_dqn(env, direction_model, duration_model, num_episodes):
    total_waiting_time_dqn = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Ensure tensor is on the correct device

            # Ensure no gradients are computed
            with torch.no_grad():
                # Predict direction and duration from the respective models
                direction_logits = direction_model(state_tensor)
                duration_logits = duration_model(state_tensor)

                # Convert logits to actual action choices
                direction_action = torch.argmax(direction_logits).item()
                duration_action = torch.argmax(duration_logits).item()

            # Environment steps based on the direction and duration actions
            next_state, reward, waiting_time, done, _ = env.step(direction_action, duration_action)
            state = next_state
            total_waiting_time_dqn += waiting_time

    return total_waiting_time_dqn

def evaluate_static(env, num_episodes):
    total_waiting_time_static = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            next_state, reward, waiting_time, done, _ = env.step()  # Static method
            state = next_state
            total_waiting_time_static += waiting_time
    return total_waiting_time_static

# Load environments
env_dqn = SumoEnv(r'xml\sumo_config.sumocfg', max_steps=1000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DQN model
def load_full_model(model_path):
    model = torch.load(model_path, map_location=device)  # Ensure model is loaded to the correct device
    model.eval()
    return model

direction_agent = load_full_model('direction_agent_model_full2.pth')
duration_agent = load_full_model('duration_agent_model_full2.pth')

direction_agent.to(device)
duration_agent.to(device)

# Run evaluations
waiting_time_dqn = evaluate_dqn(env_dqn, direction_agent, duration_agent, 10)
env_dqn.close()

env_static = SumoEnv2(r'xml\sumo_config.sumocfg', max_steps=1000)
waiting_time_static = evaluate_static(env_static, 10)
env_static.close()

# Calculate reduction
def plot_reduction(wait_time_static, wait_time_dqn):
    reduction_percent = ((wait_time_static - wait_time_dqn) / wait_time_static) * 100
    plt.figure(figsize=(10, 6))
    plt.bar(['Static Control', 'DQN Control'], [wait_time_static, wait_time_dqn], color=['blue', 'green'])
    plt.xlabel('Control Type')
    plt.ylabel('Total Waiting Time')
    plt.title('Comparison of Total Waiting Time: Static vs DQN Control')
    plt.show()

    return reduction_percent

reduction = plot_reduction(waiting_time_static, waiting_time_dqn)
print(f"Reduction in waiting time by using DQN over static control: {reduction:.2f}%")
