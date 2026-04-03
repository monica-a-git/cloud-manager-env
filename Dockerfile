FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Hugging Face Spaces MANDATES port 7860
CMD["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]