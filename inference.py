import os
import asyncio
import json
from typing import List, Dict, Any
from openai import OpenAI
from openenv.core.generic_client import GenericEnvClient

# STRICT: Use environment variables injected by the evaluation system
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "CloudManagerEnv"

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
IMAGE_NAME = LOCAL_IMAGE_NAME if LOCAL_IMAGE_NAME else "cloud-env:latest"

ENV_SERVER_URL = os.getenv("ENV_SERVER_URL") # if deployed to HF
TASKS = ["cloud-management-easy", "cloud-management-medium", "cloud-management-hard"]
MAX_STEPS = 30
TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.8  # 80% score threshold

def get_model_message(client: OpenAI, step: int, last_obs: Dict[str, Any], last_reward: float, history: List[str], task_name: str) -> Dict[str, Any]:
    sys_prompt = f"You are an automated Cloud AI infrastructure manager. Task: {task_name}. You must scale servers dynamically with unpredictable traffic spikes and drops to prevent crashes while minimizing costs. Maximize uptime efficiency."
    user_prompt = f"Current Obs: {json.dumps(last_obs)}. Last Reward: {last_reward}. History: {history}. Provide your action strictly in JSON format. Example: {{\\\"target_server_id\\\": \\\"web-2\\\", \\\"command\\\": \\\"start\\\"}}. You can use 'start', 'stop', or 'none'. Target IDs: web-1, web-2, web-3, db-1."
    
    try:
        import re
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt + "\nOutput STRICTLY only raw valid JSON without markdown backticks."}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text) if text else {"target_server_id": "none", "command": "none"}
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"target_server_id": "none", "command": "none"}

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: Any = None):
    print(f"[STEP] step={step} reward={reward} action={action} done={done} error={error}", flush=True)

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]):
    print(f"[END] task={task} score={score} steps={steps}", flush=True)

async def run_task(client: OpenAI, task_name: str) -> None:
    env = None
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        if ENV_SERVER_URL:
            env = GenericEnvClient(base_url=ENV_SERVER_URL)
        else:
            try:
                env = await GenericEnvClient.from_docker_image(
                    IMAGE_NAME, 
                    env_vars={
                        "TASK_NAME": task_name,
                        "API_BASE_URL": API_BASE_URL,
                        "API_KEY": API_KEY,
                        "MODEL_NAME": MODEL_NAME
                    }
                )
            except Exception as e:
                print(f"[ERROR] Docker startup failed for {task_name}: {e}", flush=True)
                return

        try:
            result = await env.reset() 
            last_obs = result.observation
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action_data = get_model_message(client, step, last_obs, last_reward, history, task_name)

                result = await env.step(action_data)
                obs = result.observation

                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                steps_taken = step
                last_obs = obs
                last_reward = reward

                log_step(step=step, action=action_data, reward=reward, done=done)
                history.append(f"Step {step}: {action_data} -> reward {reward:+.2f}")

                if done:
                    final_info = obs.get("metadata", {}).get("final_info", {})
                    score = final_info.get("normalized_score", 0.0)
                    break

            if score == 0.0:
                score = sum(rewards) / (MAX_STEPS * 1.0) # approx normalization
            
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as e:
            print(f"[ERROR] Exception during task execution: {e}", flush=True)
        
    finally:
        if env:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    if not API_BASE_URL or not API_KEY:
        raise RuntimeError("Missing required injected environment variables: API_BASE_URL and API_KEY")

    # STRICT: Follow the 'HOW TO FIX' instructions exactly for evaluation
    base_url = API_BASE_URL
    api_key = API_KEY
    
    openai_client = OpenAI(base_url=base_url, api_key=api_key)

    for task in TASKS:
        await run_task(openai_client, task)

if __name__ == "__main__":
    asyncio.run(main())
