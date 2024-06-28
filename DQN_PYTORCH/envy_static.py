import gym
import numpy as np
from gym import spaces
import traci
from sumolib import checkBinary
import random
class SumoEnv2(gym.Env):
    def __init__(self, sumo_cfg, max_steps=10000):
        super(SumoEnv2, self).__init__()
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.current_step = 0
        self.intersection_id = "J3"
        self.current_direction = 0
        self.current_phase_duration = 30  # Set fixed phase duration
        self.observation_space = spaces.Box(low=0, high=255, shape=(36,), dtype=np.float32)

        sumo_binary = checkBinary('sumo')
        traci.start([sumo_binary, "-c", self.sumo_cfg])

    def reset(self):
        traci.load(["-c", self.sumo_cfg])
        self.current_step = 0
        self.current_phase_duration = 30 #random.randint(10, 50) 
        self.current_direction = -1  # Reset to cycle from the start
        return self._get_state()
        #random.randint(20, 50) 

    def step(self):
        self.current_direction = (self.current_direction + 1) % 4
        self._apply_action()
        for _ in range(self.current_phase_duration):
            traci.simulationStep()
            self.current_step += 1

        state = self._get_state()
        reward = self._get_reward()
        waiting_time = self._calculate_total_waiting_time()  # Calculate total waiting time
        remaining_vehicles = traci.vehicle.getIDCount()
        done = remaining_vehicles == 0 or self.current_step >= self.max_steps
       # done = self.current_step >= self.max_steps
        return state, reward, waiting_time, done, {}  # Return waiting time as part of the output

    def _apply_action(self):
        directions = ["GGGGGrrrrrrrrrrrrrrr", "rrrrrGGGGgrrrrrrrrrr", "rrrrrrrrrrgGgggrrrrr", "rrrrrrrrrrrrrrrggggg"]
        try:
            traci.trafficlight.setRedYellowGreenState(self.intersection_id, directions[self.current_direction])
        except traci.exceptions.TraCIException as e:
            print(f"TraCIException: {e}")
            traci.close()

    def _get_state(self):
        # Implement state collection here
        return np.random.rand(36)

    def _get_reward(self):
        # Implement reward calculation here
        return -np.random.rand()

    def _calculate_total_waiting_time(self):
        total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
        return total_waiting_time
    def close(self):
        traci.close()

