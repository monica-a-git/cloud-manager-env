from pydantic import BaseModel
from typing import List, Optional
from openenv.core.env_server.types import Action as OpenEnvAction, Observation as OpenEnvObservation, State as OpenEnvState

class ServerData(BaseModel):
    server_id: str
    capacity: int
    cost_per_step: float
    is_active: bool

class Observation(OpenEnvObservation):
    step_number: int
    current_traffic: int
    active_servers: List[ServerData]
    inactive_servers: List[ServerData]
    history_log: List[str]

class Action(OpenEnvAction):
    target_server_id: str
    command: str  # "start", "stop", or "none"

class State(OpenEnvState):
    max_steps: int
    servers: List[ServerData]
    total_cost: float
    total_crashes: int
    traffic_profile: List[int]
    total_reward: float