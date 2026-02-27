FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/

# Models directory will be populated by CI or mounted at runtime
RUN mkdir -p models metrics

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from src.deploy.health_check import check_health; exit(0 if check_health() else 1)"

EXPOSE 8000

CMD ["python", "-m", "src.main"]
