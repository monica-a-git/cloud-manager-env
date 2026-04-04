import gradio as gr
from server.inference import run_simulation

print("APP STARTED")

with gr.Blocks() as demo:
    gr.Markdown("# ☁️ AI Cloud Server Manager")

    difficulty = gr.Dropdown(
        choices=["Easy", "Medium", "Hard"],
        value="Medium",
        label="Difficulty"
    )

    start_btn = gr.Button("Start")

    logs = gr.Textbox(label="Logs", lines=10)
    stats = gr.Textbox(label="Stats")
    live = gr.Textbox(label="Live")
    grade = gr.Textbox(label="Final Grade")

    start_btn.click(
        fn=run_simulation,
        inputs=[difficulty],
        outputs=[logs, stats, live, grade]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)