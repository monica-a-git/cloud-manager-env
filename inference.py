import os
import asyncio
import json
from typing import List, Dict, Any
from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
IMAGE_NAME = LOCAL_IMAGE_NAME if LOCAL_IMAGE_NAME else "cloud-env:latest"

ENV_SERVER_URL = os.getenv("ENV_SERVER_URL") # if deployed to HF
TASKS = ["cloud-management-easy", "cloud-management-medium", "cloud-management-hard"]
MAX_STEPS = 30
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.8  # 80% score threshold

def get_model_message(client: OpenAI, step: int, last_obs: Dict[str, Any], last_reward: float, task_name: str) -> Dict[str, Any]:
    sys_prompt = f"You are an automated Cloud AI infrastructure manager. Task: {task_name}. You must scale servers dynamically with unpredictable traffic spikes and drops to prevent crashes while minimizing costs. Maximize uptime efficiency."
    user_prompt = f"Current Obs: {json.dumps(last_obs)}. Last Reward: {last_reward}. Provide your action strictly in JSON format. Example: {{\\\"target_server_id\\\": \\\"web-2\\\", \\\"command\\\": \\\"start\\\"}}. You can use 'start', 'stop', or 'none'. Target IDs: web-1, web-2, web-3, db-1."
    
    try:
        import re
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt + "\nOutput STRICTLY only raw valid JSON without markdown backticks."}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text) if text else {"target_server_id": "none", "command": "none"}
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"target_server_id": "none", "command": "none"}

def log_start(task: str, env: str, model: str):
    print(f"\n==========================================")
    print(f"[{task}] STARTING EVAL | Env: {env} | Model: {model}")
    print(f"==========================================")

def log_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: Any):
    action_str = f"{action.get('command')} -> {action.get('target_server_id')}"
    print(f"Step {step:02d} | Action: {action_str:20} | Reward: {reward:+.1f} | Done: {done}")

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(f"--- Episode Ended ---")
    print(f"Success: {'YES' if success else 'NO'}")
    print(f"Score: {score:.3f} / 1.000")
    print(f"Steps: {steps}")
    print(f"Total Trajectory Reward: {sum(rewards)}")

async def run_task(client: OpenAI, task_name: str) -> None:
    if ENV_SERVER_URL:
        # Connecting directly to a running HF space or local FastAPI
        env = GenericEnvClient(base_url=ENV_SERVER_URL)
    else:
        # Spin up a docker container for isolated evaluation
        env = await GenericEnvClient.from_docker_image(IMAGE_NAME, env_vars={"TASK_NAME": task_name})

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env="CloudManagerEnv", model=MODEL_NAME)

    try:
        # Note: GenericEnvClient reset accepts arbitrary dict.
        result = await env.reset() # This returns a StepResult with observation
        last_obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_data = get_model_message(client, step, last_obs, last_reward, task_name)

            result = await env.step(action_data)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None # from generic client if errors occur they raise exceptions usually

            rewards.append(reward)
            steps_taken = step
            last_obs = obs
            last_reward = reward

            log_step(step=step, action=action_data, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_data} -> reward {reward:+.2f}")

            if done:
                # The final metadata from CloudManagerEnv will hold the normalized score
                final_info = obs.get("metadata", {}).get("final_info", {})
                score = final_info.get("normalized_score", 0.0)
                break

        # Override score with fallback calculation if missing from metadata
        if score == 0.0:
            score = sum(rewards) / (MAX_STEPS * 2.0) # approx max total reward
            score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    if not HF_TOKEN:
        print("Set HF_TOKEN environment variable. Proceeding with dummy key for debug syntax check...")

    openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    for task in TASKS:
        await run_task(openai_client, task)

if __name__ == "__main__":
    asyncio.run(main())
