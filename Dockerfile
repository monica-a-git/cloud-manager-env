FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# VERY IMPORTANT → HF expects port 7860
ENV PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]