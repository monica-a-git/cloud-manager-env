from pydantic import BaseModel
from typing import List

class Server(BaseModel):
    server_id: str
    capacity: int
    cost_per_step: float
    is_active: bool

class Observation(BaseModel):
    step_number: int
    current_traffic: int
    active_servers: List[Server]
    inactive_servers: List[Server]
    history_log: List[str]

class Action(BaseModel):
    target_server_id: str
    command: str  # "start", "stop", or "none"

class Reward(BaseModel):
    value: float
    reason: str

class State(BaseModel):
    step_number: int
    max_steps: int
    servers: List[Server]
    total_cost: float
    total_crashes: int
    traffic_profile: List[int]