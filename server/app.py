import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from server.my_env_environment import CloudManagerEnv
from models import Action
import uvicorn

app = FastAPI(title="Cloud Manager Environment API")

# Global environment instance
env = None

class StepRequest(BaseModel):
    target_server_id: str
    command: str

@app.post("/reset")
def reset(difficulty: str = "Medium"):
    global env
    env = CloudManagerEnv(difficulty=difficulty)
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: StepRequest):
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    
    action_obj = Action(target_server_id=action.target_server_id, command=action.command)
    obs, reward, done, info = env.step(action_obj)
    
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)