# Import necessary libraries and classes
from enhance_envi import SumoEnv  # Ensure this is correctly pointing to your SumoEnv class
from dqn_agent import DQNAgent  # Ensure this points to your DQNAgent class
import torch
# Setup the SUMO environment and DQN agents
env = SumoEnv(r'DQN_PYTORCH\xml\sumo_config.sumocfg', max_steps=2000)
state_size = 22  # As defined by your environment's observation space
direction_agent = DQNAgent(state_size, 4)  # 4 possible directions
duration_agent = DQNAgent(state_size, 6)  # 6 possible durations (10, 20, 30, 40, 50, 60 seconds)

# Training parameters
num_episodes = 100  # Number of episodes to train
batch_size = 64  # Size of batch taken from replay buffer

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Agents decide on actions
        direction_action = direction_agent.act(state)
        duration_action = duration_agent.act(state)
        
        # Environment steps based on actions
        next_state, reward, done, _ = env.step(direction_action, duration_action)
        
        # Save experiences in replay memory
        direction_agent.remember(state, direction_action, reward, next_state, done)
        duration_agent.remember(state, duration_action, reward, next_state, done)
        
        # Sample from memory and learn
        if len(direction_agent.memory) > batch_size:
            direction_agent.replay()
        if len(duration_agent.memory) > batch_size:
            duration_agent.replay()
        
        # Update state and accumulate reward
        state = next_state
        total_reward += reward
    if (episode + 1) % 25 == 0:
        direction_agent.update_target_model()
        duration_agent.update_target_model()
        torch.save(direction_agent.model.state_dict(), 'direction_agent_model7.pth')
        torch.save(direction_agent.model, 'direction_agent_model_full7.pth')

        torch.save(duration_agent.model.state_dict(), 'duration_agent_model7.pth')
        torch.save(duration_agent.model, 'duration_agent_model_full7.pth')
    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

torch.save(direction_agent.model.state_dict(), 'direction_agent_model7.pth')
torch.save(direction_agent.model, 'direction_agent_model_full7.pth')

torch.save(duration_agent.model.state_dict(), 'duration_agent_model7.pth')
torch.save(duration_agent.model, 'duration_agent_model_full7.pth')

print("Training completed!")
