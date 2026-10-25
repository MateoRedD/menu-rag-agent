# Same Python version as local development.
FROM python:3.14.6-slim

# Unbuffered output so logs show up immediately in `docker logs` and Azure.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Dependencies first: this layer is cached and only rebuilt when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs. Secrets are NOT copied: they come as environment variables at run time.
COPY data/ data/
COPY rag.py menu.py tools.py agent.py api.py ./

# Don't run as root inside the container.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
