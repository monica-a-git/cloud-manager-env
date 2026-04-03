FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Tell Python to run the app.py located inside the server folder!
CMD ["python", "server/inference.py"]