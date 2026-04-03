FROM python:3.9-slim
WORKDIR /app
# Upgrade pip to handle dependency resolutions better
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "inference.py"]