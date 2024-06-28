import torch
from enhance_envi import SumoEnv  # Ensure this is correctly pointing to your SumoEnv class
from dqn_agent import DQNAgent  # Ensure this points to your DQNAgent class

# Function to load a model
def load_full_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

# Evaluate the performance of both models
def evaluate_model(env, direction_model, duration_model, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                direction_action = torch.argmax(direction_model(state_tensor)).item()
                duration_action = torch.argmax(duration_model(state_tensor)).item()
                print('direction_action: ', direction_action)
                print('duration_action: ', (duration_action + 1) * 10)
                
            next_state, reward, done, _ = env.step(direction_action, duration_action)
            state = next_state
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Initialize your environment
env = SumoEnv('xml\sumo_config.sumocfg', max_steps=1000)

# Load models
state_size = 22  # Adjust based on your environment's observation space
direction_model = load_full_model('direction_agent_model_full6.pth')  # Assuming 4 direction actions
duration_model = load_full_model('duration_agent_model_full6.pth')  # Assuming 6 duration options

# Run evaluation
num_episodes = 10  # Define how many episodes to run for the evaluation
evaluate_model(env, direction_model, duration_model, num_episodes)
