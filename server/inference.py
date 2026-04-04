import os
import json
import time
from openai import OpenAI
from models import Action
from server.my_env_environment import CloudManagerEnv

API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME")

print("Inference loaded")

system_prompt = """
You are a Cloud Infrastructure Manager.
GOAL: Keep capacity slightly above traffic.

Return JSON:
{"target_server_id": "web-2", "command": "start"}
"""

def run_simulation(difficulty):
    print("Simulation started")

    if not all([API_BASE_URL, API_KEY, MODEL_NAME]):
        yield "Missing API config", "", "", "ERROR"
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = CloudManagerEnv(difficulty=difficulty)
    obs = env.reset()

    log_text = "Starting...\n"

    for step in range(env.max_steps):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Observation: {obs.model_dump_json()}"}
                ],
                temperature=0.0
            )
            action_data = json.loads(response.choices[0].message.content)
        except Exception as e:
            print("LLM ERROR:", e)
            action_data = {"target_server_id": "none", "command": "none"}

        action = Action(**action_data)

        obs, reward, done, info = env.step(action)

        traffic = env.traffic_profile[step]
        capacity = sum(s.capacity for s in env.servers if s.is_active)

        log_text = reward.reason + "\n" + log_text
        stats = f"Crashes: {env.total_crashes} | Cost: ${env.total_cost:.2f}"

        yield log_text, stats, f"T:{traffic} C:{capacity}", "Running"

        time.sleep(0.1)   # VERY IMPORTANT

        if done:
            break

    yield log_text, stats, "Done", f"Grade: {info.get('grade')}"