FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Run the shell script
CMD ["./start.sh"]