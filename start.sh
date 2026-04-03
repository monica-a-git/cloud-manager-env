#!/bin/bash

# 1. Start the FastAPI server in the background
uvicorn server.app:app --host 0.0.0.0 --port 7860 &

# 2. Wait a few seconds to ensure the server is fully awake
sleep 5

# 3. Set the SPACE_URL to localhost since they are in the same container now
export SPACE_URL="http://127.0.0.1:7860"

# 4. Run your AI client script
echo "Starting AI Inference..."
python inference.py

# 5. Keep the container alive by waiting on the background Uvicorn process
wait