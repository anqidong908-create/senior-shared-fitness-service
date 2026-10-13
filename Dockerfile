FROM python:3.12-slim

WORKDIR /app
COPY requirements.lock requirements.txt ./
RUN pip install --no-cache-dir -r requirements.lock
COPY . .

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV DEVICE_DATA_MODE=simulator

EXPOSE 8010
CMD ["uvicorn", "multi_agent_service:app", "--host", "0.0.0.0", "--port", "8010"]
