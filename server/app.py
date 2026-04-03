from fastapi import FastAPI, HTTPException
from models import Action, Server
from server.my_env_environment import CloudManagerEnv

app = FastAPI()

# Default to "Medium" difficulty for the API start
default_servers =[
    Server(server_id="web-1", capacity=100, cost_per_step=10.0, is_active=True),
    Server(server_id="web-2", capacity=100, cost_per_step=10.0, is_active=False)
]
env = CloudManagerEnv(servers=default_servers, traffic_profile=[80, 150, 150])

@app.post("/reset")
def reset_environment():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step_environment(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))