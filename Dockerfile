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

ENV PORT=8000
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]