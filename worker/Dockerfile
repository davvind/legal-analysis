# SYN worker v4 — Container Apps Job image
FROM python:3.11-slim

WORKDIR /app

# psycopg2-binary + azure-sdk trenger ikke build-tools; slim holder.
COPY worker_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY worker.py .

# I Container Apps Job settes WORKER_MODE=once via env: hver jobb-eksekvering
# tar ÉN melding og avslutter. KEDA skalerer antall eksekveringer etter kølengde.
CMD ["python", "worker.py"]
