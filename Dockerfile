FROM python:3.11-slim

# Fail fast and stream logs instead of buffering them (so container logs are useful).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies are copied and installed first so a source-only change doesn't
# invalidate the (slow) pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build-time: regenerate corpora, train tokenizer + tiny LM + safety classifier
# so the image ships with working checkpoints out of the box. In a real deployment
# these steps would instead be an offline training job whose output artifacts
# (tokenizer/*.model, model/*.pt, classifier/*.pt) are pulled from a model registry
# (e.g. S3 / MLflow / Hugging Face Hub) rather than trained inside the image build.
RUN python -m data.generate_corpus \
    && python -m tokenizer.train_sentencepiece \
    && python -m model.train \
    && python -m classifier.generate_labeled_data \
    && python -m classifier.train

# Run as an unprivileged user: a compromised web process should not own /app.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
