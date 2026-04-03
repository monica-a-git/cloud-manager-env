import os
import json
import asyncio
import httpx
from typing import List
from openai import OpenAI

# ==========================================
# CONFIGURATION
# ==========================================
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME")

# Put your Hugging Face Space URL here (e.g., https://a-monica-cloud-manager.hf.space)
SPACE_URL = os.environ.get("SPACE_URL", "http://127.0.0.1:7860") 

MAX_STEPS = 10
TASK_NAME = "Cloud_Server_Manager"
TEMPERATURE = 0.0
MAX_TOKENS = 150

SYSTEM_PROMPT = """
You are an automated Cloud Infrastructure Manager. 
Your goal is to handle incoming traffic by starting and stopping servers.
- If active server capacity is LESS than traffic, the site crashes.
- If active server capacity is WAY HIGHER than traffic, you waste money.

You MUST output ONLY a valid JSON object exactly like this:
{
    "target_server_id": "web-1", 
    "command": "start" 
}
Valid commands are "start", "stop", or "none".
"""

# ==========================================
# MOCK OPENENV ASYNC WRAPPER
# ==========================================
class EnvResult:
    def __init__(self, observation, reward, done):
        self.observation = observation
        self.reward = reward
        self.done = done

class AsyncCloudEnv:
    """Wraps your FastAPI endpoints to match the exact async OpenEnv style you requested."""
    def __init__(self, url: str):
        self.url = url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def reset(self) -> EnvResult:
        resp = await self.client.post(f"{self.url}/reset")
        resp.raise_for_status()
        return EnvResult(observation=resp.json(), reward=0.0, done=False)

    async def step(self, message_json: str) -> EnvResult:
        try:
            action_data = json.loads(message_json)
        except json.JSONDecodeError:
            action_data = {"target_server_id": "none", "command": "none"}
            
        resp = await self.client.post(f"{self.url}/step", json=action_data)
        resp.raise_for_status()
        data = resp.json()
        return EnvResult(
            observation=data["observation"], 
            reward=data["reward"]["value"], 
            done=data["done"]
        )

    async def close(self):
        await self.client.aclose()


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def log_start(task, env, model):
    print(f"\n🚀 STARTING TASK: {task} | Env: {env} | Model: {model}")

def log_step(step, action, reward, done, error):
    print(f"   -> Step {step} | Action: {action} | Reward: {reward:+.2f} | Done: {done}")

def log_end(success, steps, score, rewards):
    print(f"✅ END | Success: {success} | Steps: {steps} | Total Score: {score:.2f}")

def build_user_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-3:]) if history else "No history yet."
    return (
        f"Observation: {json.dumps(obs)}\n"
        f"Last Reward: {last_reward}\n\n"
        f"Previous steps:\n{history_block}\n\n"
        "Send your next message as JSON."
    ).strip()

def get_model_message(client: OpenAI, step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"target_server_id": "none", "command": "none"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"target_server_id": "none", "command": "none"}'


# ==========================================
# MAIN LOOP
# ==========================================
async def main() -> None:
    if not all([API_BASE_URL, API_KEY, MODEL_NAME]):
        raise ValueError("Missing API variables (API_BASE_URL, HF_TOKEN, MODEL_NAME)")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = AsyncCloudEnv(url=SPACE_URL)

    history: List[str] = []
    rewards: List[float] =[]
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=SPACE_URL, model=MODEL_NAME)

    try:
        result = await env.reset() 
        last_obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # 1. Get action from AI
            message = get_model_message(client, step, last_obs, last_reward, history)

            # 2. Step the remote environment
            result = await env.step(message)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_obs = obs
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards)
        success = score > 0.0 

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())