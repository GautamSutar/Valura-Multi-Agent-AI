FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8080
# The grading network's only route out is the gateway; Agno's default
# telemetry call to os-api.agno.com would otherwise try to reach the open
# internet on every agent run and hang or fail there.
ENV AGNO_TELEMETRY=false
EXPOSE 8080

CMD python -m uvicorn takehome.service:app --host 0.0.0.0 --port ${PORT:-8080}
