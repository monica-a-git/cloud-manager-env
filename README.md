---
title: Cloud AI Manager OpenEnv
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
app_port: 8000
---

# Cloud AI Manager OpenEnv ☁️

Welcome to **Cloud AI Manager**, a real-world OpenEnv simulation where an AI agent acts as a Cloud Infrastructure Manager. In this environment, the agent must monitor and predict rapidly fluctuating Web Traffic and scale cloud resources efficiently—matching capacity dynamically to minimize costs, prevent catastrophic crashes, and optimize stability over the long run!

**Deployed on :** https://huggingface.co/spaces/a-monica/cloud-manager

## Motivations

This environment replicates the real-world Site Reliability Engineering (SRE) and auto-scaling task where humans typically monitor Grafana dashboards and spin up/down cloud instances based on volatile usage. The agent learns resource conservation vs. availability trade-offs, handling realistic noise, sudden virality spikes, and capacity planning.

## Action Space

The agent can output JSON commands predicting which servers to target. The valid JSON strict structure expected:
```json
{
  "target_server_id": "web-2",
  "command": "start"
}
```

*   `target_server_id`: (string) The ID of the server. Valid servers are `web-1` (Cap: 100), `web-2` (Cap: 100), `web-3` (Cap: 200), `db-1` (Cap: 500).
*   `command`: (string) Operates the server. Valid commands are `"start"`, `"stop"`, or `"none"`.

## Observation Space

The observation space is a full JSON payload defined strictly under typed `Pydantic` OpenEnv parameters containing:
*   `step_number`: (int) The current time-step
*   `current_traffic`: (int) Current simulated traffic value hitting the cloud load balancers.
*   `active_servers`: (List) The list of active servers, their capacity, and cost-per-step.
*   `inactive_servers`: (List) Spared servers ready to boot.
*   `history_log`: (List) Short textual log history of recent events.
*   `done`: (bool) Episode completion.
*   `reward`: (float) Reward generated per step based on capacity matched efficiently over traffic.
*   `metadata`: (dict) OpenEnv dict providing final Grade scores and stats.

## Tasks & Difficulties

This environment offers 3 scaling tasks out-of-the-box graded rigorously on 30 steps deterministically:

1. **`cloud-management-easy`**: Standard recurring traffic cycles that drop and rise smoothly on a daily sine wave without massive noise.
2. **`cloud-management-medium`**: Standard cyclic traffic paired with moderate bumpiness and unpredictability mimicking realistic hourly workloads.
3. **`cloud-management-hard`**: Extremely volatile workloads carrying brutal, sudden "viral" spikes exceeding 5X normal volume in a single step, demanding proactive scale planning.

**(Scores 0.0 – 1.0 bounded logic directly mapped via Efficiency vs Uptime calculation under the hood!)**

## Prerequisites and Setup

1. **Python 3.10+** (Tested on 3.11)
2. OpenEnv Core Framework `pip install openenv-core`
3. OpenAI module `pip install openai httpx gradio`

## Testing Locally (Baseline & UI)

You can run the interactive **Evaluation Script** locally passing your API configurations in order to validate scores and execution logic:
```bash
API_KEY="hf..." MODEL_NAME="gpt-4o-mini" python inference.py
```

Want to view it visually? Use the Gradio App!
```bash
API_KEY="hf..." python app.py
```

## Docker Deploy (Hugging Face)

It uses standard OpenEnv integration so you can push directly to a HF Space (OpenEnv SDK) via:
```bash
pip install huggingface_hub
huggingface-cli login
openenv push --repo-id user-name/cloud-manager
```
This pushes your environment automatically configured perfectly to test multi-model inference against the OpenEnv spec.


## AI AGENT SECIFICATIONS

The AI Agent should be a Text generation - chat based model. It should be able to understand the environment and act accordingly. It should be able to generate JSON output in the format specified by the environment. 