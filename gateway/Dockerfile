FROM python:3.12-slim
WORKDIR /app
COPY llm_gateway.py .
ENV GATEWAY_PORT=8600
EXPOSE 8600
CMD ["python", "llm_gateway.py"]
