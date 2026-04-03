import os
import json
import requests
from openai import OpenAI
from models import Action

api_base_url = os.environ.get("API_BASE_URL")
model_name = os.environ.get("MODEL_NAME")
hf_token = os.environ.get("HF_TOKEN")
# Address where your app.py is running
ENV_SERVER_URL = os.environ.get("ENV_SERVER_URL", "http://127.0.0.1:8000") 

if not all([api_base_url, model_name, hf_token]):
    raise ValueError("Missing API_BASE_URL, MODEL_NAME, or HF_TOKEN")

client = OpenAI(base_url=api_base_url, api_key=hf_token)

def run_agent():
    print("=============================")
    print(" STARTING CLOUD MANAGER AGENT")
    print("=============================")
    
    # 1. Reset the remote environment
    response = requests.post(f"{ENV_SERVER_URL}/reset")
    obs = response.json()
    done = False
    
    system_prompt = """
    You are an automated Cloud Infrastructure Manager. 
    Your goal is to handle incoming traffic by starting and stopping servers.
    - If active server capacity is LESS than traffic, the site crashes.
    - If active server capacity is WAY HIGHER than traffic, you waste money.
    
    You must output a JSON object exactly like this:
    {
      "target_server_id": "web-1", 
      "command": "start" 
    }
    Valid commands: "start", "stop", "none".
    """

    while not done:
        user_prompt = f"Current Observation: {json.dumps(obs)}"
        
        # 2. Get AI Decision
        llm_response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"}, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        raw_ai_response = llm_response.choices[0].message.content
        try:
            action_data = json.loads(raw_ai_response)
        except json.JSONDecodeError:
            action_data = {"target_server_id": "none", "command": "none"}
            
        action = Action(
            target_server_id=action_data.get("target_server_id", "none"),
            command=action_data.get("command", "none")
        )
        
        print(f"\n[Step {obs.get('step_number')}] Traffic: {obs.get('current_traffic')}")
        print(f"AI decides to: {action.command} on '{action.target_server_id}'")
        
        # 3. Send action to the remote environment
        step_res = requests.post(
            f"{ENV_SERVER_URL}/step", 
            json=action.model_dump()
        ).json()
        
        obs = step_res["observation"]
        reward = step_res["reward"]
        done = step_res["done"]
        info = step_res["info"]
        
        print(f"Result -> Reward: {reward['value']} | Reason: {reward['reason']}")
        
    print("\n--- Task Finished ---")
    print(f"Final Grade: {info.get('final_grade')} / 1.0")
    print(f"Total Cost: ${info.get('total_cost')}")

if __name__ == "__main__":
    run_agent()