from openenv.core.env_server import create_app
from my_env.server.my_env_environment import CloudManagerEnv
from my_env.models import Action, Observation
from my_env.app import demo
import gradio as gr

base_app = create_app(
    CloudManagerEnv,
    Action,
    Observation,
    max_concurrent_envs=10
)

app = gr.mount_gradio_app(base_app, demo, path="/")

# For running outside of openenv if needed
def main():
    import uvicorn
    uvicorn.run("my_env.server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
