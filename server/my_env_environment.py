from my_env.models import Server, Observation, Action, Reward, State
from typing import List, Tuple

class CloudManagerEnv:
    def __init__(self, servers: List[Server], traffic_profile: List[int]):
        self.initial_servers =[s.model_copy() for s in servers]
        self.traffic_profile = traffic_profile
        self.max_steps = len(traffic_profile)
        self.reset()

    def reset(self) -> Observation:
        self.servers = [s.model_copy() for s in self.initial_servers]
        self.step_number = 0
        self.total_cost = 0.0
        self.total_crashes = 0
        self.history =[]
        return self._get_obs()

    def state(self) -> State:
        return State(
            step_number=self.step_number,
            max_steps=self.max_steps,
            servers=self.servers,
            total_cost=self.total_cost,
            total_crashes=self.total_crashes,
            traffic_profile=self.traffic_profile
        )

    def _get_obs(self) -> Observation:
        active =[s for s in self.servers if s.is_active]
        inactive = [s for s in self.servers if not s.is_active]
        current_traffic = self.traffic_profile[self.step_number] if self.step_number < self.max_steps else 0

        return Observation(
            step_number=self.step_number,
            current_traffic=current_traffic,
            active_servers=active,
            inactive_servers=inactive,
            history_log=self.history[-3:] 
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        if self.step_number >= self.max_steps:
            return self._get_obs(), Reward(value=0.0, reason="Episode finished."), True, {}

        current_traffic = self.traffic_profile[self.step_number]
        step_reward = 0.0
        reasons =[]

        target_server = next((s for s in self.servers if s.server_id == action.target_server_id), None)
        
        if action.command != "none" and target_server is None:
            step_reward -= 0.1
            reasons.append(f"Penalty: Server {action.target_server_id} does not exist.")
        elif action.command == "start":
            if target_server.is_active:
                step_reward -= 0.05
                reasons.append("Wasted action: Server already running.")
            else:
                target_server.is_active = True
                reasons.append(f"Started server {target_server.server_id}.")
        elif action.command == "stop":
            if not target_server.is_active:
                step_reward -= 0.05
                reasons.append("Wasted action: Server already stopped.")
            else:
                target_server.is_active = False
                reasons.append(f"Stopped server {target_server.server_id}.")
        elif action.command == "none":
            reasons.append("Took no action.")
        else:
            step_reward -= 0.1
            reasons.append("Penalty: Invalid command.")

        total_capacity = sum(s.capacity for s in self.servers if s.is_active)
        step_cost = sum(s.cost_per_step for s in self.servers if s.is_active)
        self.total_cost += step_cost

        if total_capacity < current_traffic:
            self.total_crashes += 1
            step_reward -= 1.0
            reasons.append(f"CRASH! Traffic ({current_traffic}) exceeded capacity ({total_capacity}).")
        else:
            step_reward += 0.5
            reasons.append("Traffic handled smoothly.")
            wasted_capacity = total_capacity - current_traffic
            if wasted_capacity == 0:
                 step_reward += 0.5 
                 reasons.append("Perfect resource allocation.")
            elif wasted_capacity > 100:
                 step_reward -= 0.2 
                 reasons.append("Warning: Over-provisioned. Wasting money on idle servers.")
            else:
                 step_reward += 0.2 
                 reasons.append("Good buffer maintained.")

        reward_obj = Reward(value=step_reward, reason=" | ".join(reasons))
        self.history.append(f"Step {self.step_number}: {reward_obj.reason} (Cost: ${step_cost})")
        
        self.step_number += 1
        done = self.step_number >= self.max_steps
        
        info = {}
        if done:
            final_score = 1.0 - (self.total_crashes * 0.4)
            if self.total_cost > (self.max_steps * 20): 
                final_score -= 0.2
            info["final_grade"] = max(0.0, min(1.0, final_score)) 
            info["total_cost"] = self.total_cost

        return self._get_obs(), reward_obj, done, info