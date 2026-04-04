FROM python:3.9-slim
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Points to server/inference.py exactly where it sits in your image
CMD ["python", "server/inference.py"]