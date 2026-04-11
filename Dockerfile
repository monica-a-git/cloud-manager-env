FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install openenv-core
RUN pip install --no-cache-dir openenv-core>=0.2.2 openai httpx gradio

COPY . .

# Install local package
RUN pip install -e .

ENV HOST="0.0.0.0"
ENV PORT=8000
ENV PYTHONPATH=/app
EXPOSE 8000

# Shell format is required so HF Spaces can dynamically override PORT and WORKERS
CMD uvicorn my_env.server.app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers ${WORKERS:-4}