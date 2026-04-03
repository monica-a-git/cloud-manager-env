import math
import random
from models import Server, Observation, Action, Reward, State

class CloudManagerEnv:
    def __init__(self, difficulty: str = "Medium"):
        self.difficulty = difficulty.lower()
        self.max_steps = 60 # 60 steps = ~4-5 minutes in real-time visualization
        
        # Define servers based on difficulty
        self.initial_servers =[
            Server(server_id="web-1", capacity=100, cost_per_step=10.0, is_active=True),
            Server(server_id="web-2", capacity=100, cost_per_step=10.0, is_active=False),
            Server(server_id="web-3", capacity=200, cost_per_step=18.0, is_active=False),
            Server(server_id="db-1", capacity=500, cost_per_step=40.0, is_active=False),
        ]
        
        # Generate Traffic Profile
        self.traffic_profile = self._generate_traffic()
        self.reset()

    def _generate_traffic(self):
        traffic =[]
        for i in range(self.max_steps):
            # Base sine wave (simulating day/night cycle)
            base = 150 + 100 * math.sin(i / 5.0) 
            
            if self.difficulty == "easy":
                traffic.append(int(max(0, base - 50))) # Gentle traffic
            elif self.difficulty == "medium":
                traffic.append(int(max(0, base + random.randint(-50, 50)))) # Bumpy traffic
            elif self.difficulty == "hard":
                # Hard mode includes sudden massive "viral" spikes
                spike = 400 if random.random() > 0.9 else 0
                traffic.append(int(max(0, base + spike + random.randint(-50, 150))))
        return traffic

    def reset(self):
        self.servers =[s.model_copy() for s in self.initial_servers]
        self.step_number = 0
        self.total_cost = 0.0
        self.total_crashes = 0
        self.history =[]
        return self._get_obs()

    def _get_obs(self) -> Observation:
        active =[s for s in self.servers if s.is_active]
        inactive =[s for s in self.servers if not s.is_active]
        current_traffic = self.traffic_profile[self.step_number] if self.step_number < self.max_steps else 0

        return Observation(
            step_number=self.step_number,
            current_traffic=current_traffic,
            active_servers=active,
            inactive_servers=inactive,
            history_log=self.history[-3:]
        )

    def step(self, action: Action):
        if self.step_number >= self.max_steps:
            return self._get_obs(), Reward(value=0.0, reason="Done"), True, self._calculate_grade()

        current_traffic = self.traffic_profile[self.step_number]
        step_reward = 0.0
        log_msg = f"Step {self.step_number} | Traffic: {current_traffic} | "

        target_server = next((s for s in self.servers if s.server_id == action.target_server_id), None)
        
        # Action Logic
        if action.command == "start" and target_server and not target_server.is_active:
            target_server.is_active = True
            log_msg += f"Started {target_server.server_id}. "
        elif action.command == "stop" and target_server and target_server.is_active:
            target_server.is_active = False
            log_msg += f"Stopped {target_server.server_id}. "
        else:
            log_msg += "No changes. "

        # Physics & Cost Logic
        total_capacity = sum(s.capacity for s in self.servers if s.is_active)
        step_cost = sum(s.cost_per_step for s in self.servers if s.is_active)
        self.total_cost += step_cost

        # Crash vs Success logic
        if total_capacity < current_traffic:
            self.total_crashes += 1
            step_reward -= 5.0
            log_msg += f"CRASH! Capacity {total_capacity} < Traffic {current_traffic}."
        else:
            wasted = total_capacity - current_traffic
            if wasted > 200:
                step_reward -= 1.0
                log_msg += f"Warning: High waste ({wasted} unused capacity)."
            else:
                step_reward += 1.0
                log_msg += "Stable."

        self.history.append(log_msg)
        self.step_number += 1
        done = self.step_number >= self.max_steps
        
        info = self._calculate_grade() if done else {}
        return self._get_obs(), Reward(value=step_reward, reason=log_msg), done, info

    def _calculate_grade(self):
        # Meaningful Grading System
        ideal_cost = sum(self.traffic_profile) * 0.1 # Rough estimate of perfect play
        efficiency = min(100, max(0, (ideal_cost / max(1, self.total_cost)) * 100))
        uptime = ((self.max_steps - self.total_crashes) / self.max_steps) * 100

        final_score = (efficiency * 0.4) + (uptime * 0.6)

        if final_score >= 90: letter = "A"
        elif final_score >= 80: letter = "B"
        elif final_score >= 70: letter = "C"
        elif final_score >= 60: letter = "D"
        else: letter = "F"

        return {
            "crashes": self.total_crashes,
            "cost": round(self.total_cost, 2),
            "uptime_pct": round(uptime, 1),
            "efficiency_pct": round(efficiency, 1),
            "grade": letter
        }