import sys
import os
import json
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from openai import OpenAI
from models import Action  
from server.my_env_environment import CloudManagerEnv


# Fetch secrets from Hugging Face Variables
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME")

print("=== APP STARTING ===")
print("API_BASE_URL:", API_BASE_URL)
print("MODEL_NAME:", MODEL_NAME)

system_prompt = """
You are a Cloud Infrastructure Manager. 
GOAL: Keep active server capacity slightly above traffic. 
- If capacity < traffic, the site crashes (bad).
- If capacity is 200+ above traffic, you waste money (bad).

Output exactly this JSON:
{"target_server_id": "web-2", "command": "start"}
Commands: "start", "stop", "none".
"""

def run_simulation(difficulty):
    if not all([API_BASE_URL, API_KEY, MODEL_NAME]):
        yield "Error: API Keys not set in Hugging Face Settings.", "", "", "API Setup Required"
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CloudManagerEnv(difficulty=difficulty)
    env.max_steps = 30
    obs = env.reset()
    
    traffic_history = []
    capacity_history =[]
    log_text = "Starting simulation...\n"
    
    for step in range(env.max_steps):
        # 1. Ask the AI
        user_prompt = f"Observation: {obs.model_dump_json()}"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            raw_ai_response = response.choices[0].message.content
            action_data = json.loads(raw_ai_response)
        except Exception as e:
            print("LLM ERROR:", str(e))
            action_data = {"target_server_id": "none", "command": "none"}
            
        action = Action(
            target_server_id=action_data.get("target_server_id", "none"),
            command=action_data.get("command", "none")
        )

        # 2. Step the Environment
        obs, reward, done, info = env.step(action)
        
        # 3. Update Metrics for the UI
        current_traffic = env.traffic_profile[step]
        current_capacity = sum(s.capacity for s in env.servers if s.is_active)
        
        traffic_history.append(current_traffic)
        capacity_history.append(current_capacity)
        
        log_text = reward.reason + "\n" + log_text
        stats = f"Crashes: {env.total_crashes} | Cost: ${env.total_cost:.2f}"
        
        # 4. Yield live updates to the browser
        yield (
            log_text, 
            stats, 
            f"Traffic: {current_traffic} | Capacity: {current_capacity}",
            "Running..."
        )
        
        # Sleep for 4 seconds so the user can watch it happen (60 steps * 4s = 4 minutes)
        time.sleep(0.2)
        
    # 5. Final Grade
    final_grade = f"FINAL GRADE: {info['grade']} | Uptime: {info['uptime_pct']}% | Efficiency: {info['efficiency_pct']}%"
    yield log_text, stats, "Simulation Complete", final_grade

# --- UI DESIGN ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☁️ AI Cloud Server Manager Dashboard")
    gr.Markdown("Watch the AI scale servers in real-time. The simulation takes about 4 minutes.")
    
    with gr.Row():
        difficulty_dropdown = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Medium", label="Traffic Difficulty")
        start_btn = gr.Button("Start Simulation", variant="primary")
        
    with gr.Row():
        live_status = gr.Textbox(label="Live Network Status", value="Waiting to start...")
        cost_status = gr.Textbox(label="Metrics", value="Crashes: 0 | Cost: $0")
        
    grade_box = gr.Textbox(label="Final Grading Report", value="")
    logs_box = gr.Textbox(label="Live Event Logs", lines=10, max_lines=15)

    start_btn.click(
        fn=run_simulation, 
        inputs=[difficulty_dropdown], 
        outputs=[logs_box, cost_status, live_status, grade_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)