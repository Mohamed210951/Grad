import gym
import numpy as np
from gym import spaces
import traci
from sumolib import checkBinary

class SumoEnv(gym.Env):
    def __init__(self, sumo_cfg, max_steps=10000):
        super(SumoEnv, self).__init__()
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.current_step = 0
        self.intersection_id = "J3"
        self.current_direction = 0  # Initialize the current direction
        self.current_phase_duration = 10  # Initialize the current phase duration with a default value
        self.direction_space = spaces.Discrete(4)  # 4 directions
        self.duration_space = spaces.Discrete(6)  # Durations: 10, 20, 30, 40, 50, 60
        self.observation_space = spaces.Box(low=0, high=255, shape=(22,), dtype=np.float32)

        # Start SUMO in TraCI mode with GUI (for visualization; use 'sumo' for faster non-GUI mode)
        sumo_binary = checkBinary('sumo')
        traci.start([sumo_binary, "-c", self.sumo_cfg])

    def reset(self):
        traci.load(["-c", self.sumo_cfg])
        self.current_step = 0
        self.current_direction = 0  # Reset the current direction
        self.current_phase_duration = 10  # Reset the current phase duration to a default value
        return self._get_state()

    def step(self, direction_action, duration_action):
        self.current_direction = direction_action  # Update current direction based on action
        self.current_phase_duration = (duration_action + 1) * 10  # Update phase duration based on action
        self._apply_action(self.current_direction)
        for _ in range(self.current_phase_duration):
            traci.simulationStep()
            self.current_step += 1

        state = self._get_state()
        reward = self._get_reward()
        remaining_vehicles = traci.vehicle.getIDCount()
        print("Current time:", traci.simulation.getTime(), "Number of vehicles:", traci.vehicle.getIDCount())
        done = remaining_vehicles == 0 or self.current_step >= self.max_steps
        wating_time=self._calculate_total_waiting_time()
        return state, reward,wating_time, done, {}
        #return state, reward, done, {}

    def _apply_action(self, direction):
        try:
            if direction == 0:  # North
                traci.trafficlight.setRedYellowGreenState(self.intersection_id, "GGGGGrrrrrrrrrrrrrrr")
            elif direction == 1:  # East
                traci.trafficlight.setRedYellowGreenState(self.intersection_id, "rrrrrGGGGgrrrrrrrrrr")
            elif direction == 2:  # South
                traci.trafficlight.setRedYellowGreenState(self.intersection_id, "rrrrrrrrrrgGgggrrrrr")
            elif direction == 3:  # West
                traci.trafficlight.setRedYellowGreenState(self.intersection_id, "rrrrrrrrrrrrrrrggggg")
        except traci.exceptions.TraCIException as e:
            print(f"TraCIException: {e}")
            traci.close()

    def _get_state(self):
        lanes = [
            ['E1_0', 'E1_1', 'E1_2'],
            ['E2_0', 'E2_1', 'E2_2'],
            ['E3_0', 'E3_1', 'E3_2'],
            ['E4_0', 'E4_1', 'E4_2']
        ]
        
        state = []
        for direction in lanes:
            total_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in direction)
            total_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in direction)
            avg_speed = sum(traci.lane.getLastStepMeanSpeed(lane_id) for lane_id in direction) / 3
            queue_length = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in direction)
            state.extend([total_waiting_time, total_vehicles, avg_speed, queue_length])

        current_direction_vector = [0] * 4
        current_direction_vector[self.current_direction] = 1
        state.extend(current_direction_vector)
        state.append(self.current_phase_duration)
        elapsed_time = self.current_step % self.current_phase_duration if self.current_phase_duration > 0 else 0
        state.append(elapsed_time)

        return np.array(state, dtype=np.float32)

    def _get_reward(self):
        total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
        total_queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.lane.getIDList())
        
        # Calculate the number of cars that passed through the intersection
        cleared_cars = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in traci.lane.getIDList() if traci.lane.getLastStepMeanSpeed(lane) > 0)
        
        # Define rewards and penalties
        penalty_for_waiting_time = -0.1 * total_waiting_time
        penalty_for_queue_length = -0.01 * total_queue_length
        reward_for_cleared_cars = 0.05 * cleared_cars  # Adjust the factor based on desired impact

        # Calculate total reward
        reward = penalty_for_waiting_time + penalty_for_queue_length + reward_for_cleared_cars
        return reward
    def _calculate_total_waiting_time(self):
        total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
        return total_waiting_time
    def close(self):
        traci.close()
