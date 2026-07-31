# Identical base setup to the API's Dockerfile — same reasoning applies.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The web app serves on port 5000.
EXPOSE 5000

# Same gunicorn setup, bound to port 5000 for the web UI instead of the API.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
