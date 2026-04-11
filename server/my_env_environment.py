import math
import random
from typing import Optional, Any
from openenv.core.env_server import Environment
from my_env.models import Action, Observation, State, ServerData

class CloudManagerEnv(Environment[Action, Observation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True
    
    def __init__(self, task_name: str = "cloud-management-medium"):
        super().__init__()
        self.task_name = task_name.lower()
        if "easy" in self.task_name:
            self.difficulty = "easy"
        elif "hard" in self.task_name:
            self.difficulty = "hard"
        else:
            self.difficulty = "medium"

        self.max_steps = 30
        
        self.initial_servers = [
            ServerData(server_id="web-1", capacity=100, cost_per_step=10.0, is_active=True),
            ServerData(server_id="web-2", capacity=100, cost_per_step=10.0, is_active=False),
            ServerData(server_id="web-3", capacity=200, cost_per_step=18.0, is_active=False),
            ServerData(server_id="db-1", capacity=500, cost_per_step=40.0, is_active=False),
        ]
        
        self.history = []
        self._state_data = self._init_state()

    def _init_state(self):
        return State(
            step_count=0,
            max_steps=self.max_steps,
            servers=[s.model_copy() for s in self.initial_servers],
            total_cost=0.0,
            total_crashes=0,
            traffic_profile=self._generate_traffic(),
            total_reward=0.0
        )

    def _generate_traffic(self):
        traffic = []
        for i in range(self.max_steps):
            base = 150 + 100 * math.sin(i / 5.0) 
            if self.difficulty == "easy":
                traffic.append(int(max(0, base - 50)))
            elif self.difficulty == "medium":
                traffic.append(int(max(0, base + random.randint(-50, 50))))
            elif self.difficulty == "hard":
                volatile = random.choice([-200, -100, 0, 100, 300, 500]) if random.random() > 0.7 else random.randint(-50, 50)
                traffic.append(int(max(0, base + volatile)))
        return traffic

    @property
    def state(self) -> State:
        return self._state_data

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        if seed is not None:
            random.seed(seed)
        self._state_data = self._init_state()
        if episode_id is not None:
            self._state_data.episode_id = episode_id
        self.history = []
        return self._get_obs(done=False, reward=0.0)

    def _get_obs(self, done: bool, reward: float) -> Observation:
        active = [s for s in self._state_data.servers if s.is_active]
        inactive = [s for s in self._state_data.servers if not s.is_active]
        
        step_idx = min(self._state_data.step_count, self.max_steps - 1)
        current_traffic = self._state_data.traffic_profile[step_idx]

        return Observation(
            done=done,
            reward=reward,
            step_number=self._state_data.step_count,
            current_traffic=current_traffic,
            active_servers=active,
            inactive_servers=inactive,
            history_log=self.history[-3:] if hasattr(self, "history") else []
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        if self._state_data.step_count >= self.max_steps:
            return self._get_obs(done=True, reward=0.0)

        step_idx = self._state_data.step_count
        current_traffic = self._state_data.traffic_profile[step_idx]
        step_reward = 0.0
        log_msg = f"Step {step_idx} | Traffic: {current_traffic} | "

        target_server = next((s for s in self._state_data.servers if s.server_id == action.target_server_id), None)
        
        if action.command == "start" and target_server and not target_server.is_active:
            target_server.is_active = True
            log_msg += f"Started {target_server.server_id}. "
        elif action.command == "stop" and target_server and target_server.is_active:
            target_server.is_active = False
            log_msg += f"Stopped {target_server.server_id}. "
        elif action.command != "none":
            log_msg += f"Ignored {action.command} on {action.target_server_id}. "
        else:
            log_msg += "No changes. "

        total_capacity = sum(s.capacity for s in self._state_data.servers if s.is_active)
        step_cost = sum(s.cost_per_step for s in self._state_data.servers if s.is_active)
        self._state_data.total_cost += step_cost

        if total_capacity < current_traffic:
            self._state_data.total_crashes += 1
            step_reward -= 5.0
            log_msg += f"CRASH! Cap {total_capacity} < Traffic {current_traffic}."
        else:
            wasted = total_capacity - current_traffic
            if wasted > 200:
                step_reward -= 1.0
                log_msg += f"Waste ({wasted} unused)."
            else:
                step_reward += 2.0
                log_msg += "Stable."

        if not hasattr(self, 'history'):
            self.history = []
        self.history.append(log_msg)
        
        self._state_data.step_count += 1
        done = self._state_data.step_count >= self.max_steps
        self._state_data.total_reward += step_reward
        
        obs = self._get_obs(done=done, reward=step_reward)
        
        if done:
            info = self._calculate_grade()
            obs.metadata["final_info"] = info
            
        return obs

    def _calculate_grade(self):
        ideal_cost = sum(self._state_data.traffic_profile) * 0.1
        efficiency = min(100.0, max(0.0, (ideal_cost / max(1.0, self._state_data.total_cost)) * 100))
        uptime = ((self.max_steps - self._state_data.total_crashes) / self.max_steps) * 100.0

        final_score = (efficiency * 0.4) + (uptime * 0.6)
        # Ensure score is strictly between 0 and 1 exclusive (not 0.0, not 1.0)
        normalized_score = min(max(final_score / 100.0, 0.001), 0.999)
        
        if final_score >= 90: letter = "A"
        elif final_score >= 80: letter = "B"
        elif final_score >= 70: letter = "C"
        elif final_score >= 60: letter = "D"
        else: letter = "F"

        return {
            "crashes": self._state_data.total_crashes,
            "cost": round(self._state_data.total_cost, 2),
            "uptime_pct": round(uptime, 1),
            "efficiency_pct": round(efficiency, 1),
            "grade": letter,
            "normalized_score": round(normalized_score, 3)
        }