import os
import json
import time
import gradio as gr
from fastapi import FastAPI
from huggingface_hub import InferenceClient
from openenv.core.env_server import create_app
from my_env.server.my_env_environment import CloudManagerEnv
from my_env.models import Action, Observation

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME")

system_prompt = """
You are a Cloud Infrastructure Manager.
GOAL: Keep capacity slightly above traffic. Respond strictly in JSON:
{"target_server_id": "web-2", "command": "start"}
Valid commands: "start", "stop", "none".
"""

def run_simulation(difficulty):
    if not HF_TOKEN:
        yield "Missing HF_TOKEN", "", "", "ERROR"
        return

    client = InferenceClient(token=HF_TOKEN)
    task_name = f"cloud-management-{difficulty.lower()}"
    
    env = CloudManagerEnv(task_name=task_name)
    obs = env.reset()

    log_text = f"Starting {task_name}...\n"
    
    for step in range(env.max_steps):
        try:
            import re
            response = client.chat_completion(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Observation: {obs.model_dump_json()}\nOutput STRICTLY only raw valid JSON without markdown backticks."}
                ],
                temperature=0.0
            )
            text = response.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group(0)
            action_data = json.loads(text)
            action = Action(**action_data)
        except Exception as e:
            log_text = f"API FAIL: {str(e)}\n" + log_text
            print("LLM ERROR:", e)
            action = Action(target_server_id="none", command="none")

        obs = env.step(action)
        
        recent_log = obs.history_log[-1] if hasattr(obs, 'history_log') and obs.history_log else ""
        traffic = obs.current_traffic if hasattr(obs, 'current_traffic') else 0
        state = env.state
        capacity = sum(s.capacity for s in state.servers if s.is_active)

        log_text = recent_log + "\n" + log_text
        stats = f"Crashes: {state.total_crashes} | Cost: ${state.total_cost:.2f} | Reward: {state.total_reward}"

        yield log_text, stats, f"Traffic: {traffic} | Capacity: {capacity}", "Running..."

        time.sleep(0.1)

        if obs.done:
            info = obs.metadata.get("final_info", {})
            grade = info.get("grade", "N/A")
            score = info.get("normalized_score", 0.0)
            yield log_text, stats, f"Done (Traffic: 0 | Capacity: {capacity})", f"Final Grade: {grade} ({score*100}%)"
            break

with gr.Blocks() as demo:
    gr.Markdown(f"# ☁️ AI Cloud Server Manager UI\n\n**Active Model:** `{MODEL_NAME}`\n\nMonitor your AI Agent as it manages server capacity to meet unpredictable traffic spikes!")

    with gr.Row():
        difficulty = gr.Dropdown(
            choices=["Easy", "Medium", "Hard"],
            value="Medium",
            label="Task Difficulty"
        )
        start_btn = gr.Button("🚀 Start Agent Evaluation", variant="primary")

    with gr.Row():
        live = gr.Textbox(label="Live Traffic vs Capacity", placeholder="Traffic: 0 | Capacity: 0")
        stats = gr.Textbox(label="Current Stats", placeholder="Crashes: 0 | Cost: $0")
        grade = gr.Textbox(label="Status / Final Grade", placeholder="Ready")

    logs = gr.Textbox(label="Agent Action Logs", lines=15, max_lines=15)

    start_btn.click(
        fn=run_simulation,
        inputs=[difficulty],
        outputs=[logs, stats, live, grade]
    )

# 1. Create OpenEnv FastAPI base application
base_app = create_app(
    CloudManagerEnv,
    Action,
    Observation,
    max_concurrent_envs=10
)

# 2. Mount Gradio interface onto the server
app = gr.mount_gradio_app(base_app, demo, path="/")

def main():
    import uvicorn
    uvicorn.run("my_env.server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
