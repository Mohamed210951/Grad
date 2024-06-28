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
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

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
env_dqn = SumoEnv(r'DQN_PYTORCH\xml\sumo_config.sumocfg', max_steps=1500)



# Load DQN model
def load_full_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
direction_agent = load_full_model(r'DQN_PYTORCH\direction_agent_model_full-final.pth')
duration_agent = load_full_model(r'DQN_PYTORCH\duration_agent_model_full-final.pth')



  # Assume this function and path are correctly set up
  # Assume this function and path are correctly set up

# Run evaluations
waiting_time_dqn = evaluate_dqn(env_dqn, direction_agent,duration_agent, 10)
env_dqn.close()
env_static = SumoEnv2(r'DQN_PYTORCH\xml\sumo_config.sumocfg', max_steps=1500)
waiting_time_static = evaluate_static(env_static, 10)
env_static.close()
# Calculate reduction



def plot_reduction(wait_time_static, wait_time_dqn):
    reduction_percent = ((wait_time_static - wait_time_dqn) / wait_time_static) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Bar plot for waiting times
    ax1.bar(['Static Control', 'DQN Control'], [wait_time_static, wait_time_dqn], color=['green', 'red'])
    ax1.set_xlabel('Policy')
    ax1.set_ylabel('Average Waiting Time (minutes)')
    ax1.set_title('Average Waiting Time for Different Policies')

    # Bar plot for reduction percentage
    ax2.bar(['Static vs Adaptive Comparison'], [reduction_percent], color='cyan')
    ax2.set_ylabel('Percentage Reduction in Waiting Time (%)')
    ax2.set_title('Percentage Reduction in Waiting Time')

    plt.tight_layout()
    plt.show()

    return reduction_percent



reduction = plot_reduction(waiting_time_static, waiting_time_dqn)




print(f"Reduction in waiting time by using DQN over static control: {reduction:.2f}%")
