FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build-time: regenerate corpora, train tokenizer + tiny LM + safety classifier
# so the image ships with working checkpoints out of the box. In a real deployment
# these steps would instead be an offline training job whose output artifacts
# (tokenizer/*.model, model/*.pt, classifier/*.pt) are pulled from a model registry
# (e.g. S3 / MLflow / Hugging Face Hub) rather than trained inside the image build.
RUN python3 data/generate_corpus.py \
    && python3 tokenizer/train_sentencepiece.py \
    && python3 model/train.py \
    && python3 classifier/generate_labeled_data.py \
    && python3 classifier/train.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
