FROM python:3.9-slim

# Optional: best practice to set environment variables with the = syntax
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code, including the model folder
COPY . /app/

EXPOSE 8080

CMD ["python", "app.py"]
