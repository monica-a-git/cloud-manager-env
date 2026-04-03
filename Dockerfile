FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Gradio naturally runs exactly what Hugging Face wants
CMD ["python", "app.py"]