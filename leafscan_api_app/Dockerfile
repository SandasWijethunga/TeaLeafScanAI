# Use a slim official Python 3.10 image as the base — smaller footprint
# than the full python:3.10 image, while still including pip and standard
# build tools needed for TensorFlow's dependencies.
FROM python:3.11-slim

# Prevents Python from writing .pyc files to disk (unnecessary in a
# container that gets rebuilt each deploy) and forces stdout/stderr to be
# unbuffered so logs appear immediately (important for `docker logs`).
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Sets the working directory inside the container. All subsequent COPY/RUN
# commands and the final CMD execute relative to this path.
WORKDIR /app

# Copy ONLY requirements.txt first, before the rest of the source code.
# Docker caches each layer; as long as requirements.txt doesn't change,
# this pip install layer is reused on rebuilds instead of re-running,
# making iterative builds much faster.
COPY requirements.txt .

# --no-cache-dir keeps the image smaller by not storing pip's download
# cache inside the image layer.
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code (app.py, predict.py, etc.).
# This happens after the pip install layer specifically so that code
# changes don't invalidate the (slow) dependency-install cache.
COPY . .

# Documents that the container listens on port 8000. This is metadata only
# — it does not actually publish the port (that happens via docker-compose
# or `docker run -p`).
EXPOSE 8000

# Starts the API using gunicorn (a production WSGI server) instead of
# Flask's built-in development server.
#   --bind 0.0.0.0:8000   listen on all interfaces inside the container
#   --workers 1           one worker process; the TensorFlow model is
#                          loaded into memory per worker, so more workers
#                          means more RAM usage for the same model — keep
#                          this at 1 unless you've sized the container for more
#   --timeout 120         allow up to 120s per request, since the first
#                          prediction after startup can be slower due to
#                          TensorFlow's graph warm-up
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "app:app"]
