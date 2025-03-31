FROM python:3.9-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN apt-get update && apt-get install -y jq

# Copy code
COPY . /app/

EXPOSE 8080

CMD ["python", "app.py"]