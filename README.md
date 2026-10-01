# Indic Tokenizer & Decoding

A small multilingual text-generation project for **Hindi, Telugu, Malayalam, and Kannada**. It brings together tokenization, a Transformer language model, decoding strategies, sentiment filtering, and model compression in one codebase.

The project makes each stage easy to study and experiment with. You can train the model on the included synthetic data, compare how different decoding strategies behave, and use the generation pipeline from Python. A FastAPI service and Docker setup are included for exploring serving.

[Get started](#get-started) · [Architecture](#architecture) · [Results](#results) · [Project structure](#project-structure)

## What is included

| Part | Implementation |
|---|---|
| Tokenization | A handwritten BPE implementation, plus SentencePiece tokenizers used by the generation pipeline |
| Language model | TinyGPT: a decoder-only Transformer implemented in PyTorch |
| Decoding | Greedy decoding, beam search, top-k sampling, and top-p sampling |
| Sentiment filtering | A BiLSTM classifier with a generation-and-retry loop |
| Training experiments | Custom AdamW and label-smoothing implementations, with comparison scripts |
| Compression | PyTorch dynamic int8 quantization and a size/latency benchmark |
| Serving | A Python pipeline, FastAPI endpoints, and a Dockerfile |
| Evaluation | Perplexity, tokenizer efficiency, decoding diversity, and pytest tests |

The custom implementations use PyTorch for tensor operations and neural-network layers. SentencePiece handles the tokenizer used by the language model; the separate handwritten BPE implementation is useful for understanding the algorithm.

## Architecture

[![Project architecture generated with Archify](docs/architecture/tokenizer-architecture.png)](https://github.com/haarikaalla/indic-tokenizer-decoding/raw/refs/heads/main/docs/architecture/tokenizer-architecture.html)

**[Download the interactive diagram](https://github.com/haarikaalla/indic-tokenizer-decoding/raw/refs/heads/main/docs/architecture/tokenizer-architecture.html)** and open the saved HTML in your browser. Click a component to inspect its connections and source files, use **PATH** to follow a route, or try the zoom, theme, and export controls.

The viewer was generated with the official [Archify v3.0.1](https://github.com/tt-a1i/archify/tree/v3.0.1) renderer. GitHub shows the static preview; the downloaded HTML provides the interactions.

[Diagram source](docs/architecture/tokenizer-architecture.json) · [Architecture guide](docs/architecture/README.md)

## Get started

Use Python 3.11 or newer. Run the commands below from the repository root.

### 1. Set up the environment

```bash
git clone https://github.com/haarikaalla/indic-tokenizer-decoding.git
cd indic-tokenizer-decoding

python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

### 2. Prepare the data and train

```bash
python -m data.generate_corpus
python -m tokenizer.train_sentencepiece
python -m model.train
python -m classifier.generate_labeled_data
python -m classifier.train
```

These steps generate the corpora, train the tokenizers and language model, and train the sentiment classifier. Training time depends on your hardware and configuration. Language-model and classifier checkpoints are generated locally rather than committed to the repository.

### 3. Try text generation

```bash
python generate.py
python pipeline.py
```

Prompts begin with a language tag:

| Language | Tag | Example prompt |
|---|---|---|
| Hindi | `<hi>` | `<hi> राम` |
| Telugu | `<te>` | `<te> విద్యార్థి` |
| Malayalam | `<ml>` | `<ml> കുട്ടികൾ` |
| Kannada | `<kn>` | `<kn> ವಿದ್ಯಾರ್ಥಿ` |

To use the pipeline in your own Python code after training:

```python
from pipeline import TopP, build_pipeline

pipeline = build_pipeline(with_safety_filter=False)
text = pipeline.generate(
    "<te> విద్యార్థి",
    strategy=TopP(p=0.9, temperature=0.8),
    max_new_tokens=20,
    seed=0,
)
print(text)
```

Set `with_safety_filter=True` to try the sentiment-based retry policy after training the classifier. Its limitations are described below.

## Serving

The FastAPI module defines `GET /health` and `POST /generate`. It is designed to load the trained artifacts at startup and reuse them across requests.

**Current status:** `api.py` contains unresolved imports and names, including `config`, `build_pipeline`, `build_strategy`, and `logger`. These need to be fixed before the API can start. The commands below describe the serving interface; the current Docker service is affected by the same issue.

After resolving that issue and training the checkpoints:

```bash
uvicorn api:app --reload
```

In another terminal:

```bash
curl http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"<kn> ವಿದ್ಯಾರ್ಥಿ","strategy":"top_p","max_new_tokens":20,"safety_filter":false,"seed":0}'
```

The response contains `text`, `strategy`, `safety_filter_applied`, and `latency_ms`. Supported strategy names are `greedy`, `beam`, `top_k`, and `top_p`.

The Dockerfile trains the artifacts during the image build and starts Uvicorn as an unprivileged user:

```bash
docker build -t indic-generation-service .
docker run --rm -p 8000:8000 indic-generation-service
```

## Experiments and tests

Once the required artifacts have been trained, these commands let you explore individual parts of the project:

| Task | Command |
|---|---|
| Run the handwritten BPE demo | `python -m tokenizer.bpe_from_scratch` |
| Evaluate the model and decoding strategies | `python -m eval.evaluate` |
| Try sentiment-filtered generation | `python -m classifier.safety_filter_demo` |
| Compare custom training components | `python -m custom_training.benchmark` |
| Quantize the model and measure size/latency | `python -m quantization.quantize_and_benchmark` |
| Run the test suite | `python -m pytest tests/ -v` |

Tests cover tokenization, decoding, configuration, checkpoint loading, and the API contract. API integration tests require trained artifacts and are skipped when the default language-model checkpoint is absent. Running them with checkpoints also requires resolving the API issue above.

## Results

The following values were reported in the project's earlier README. They are retained as reference results, rather than a fresh benchmark of the current revision. Rerun the training and evaluation scripts to measure your own setup.

| Measurement | Reported result |
|---|---|
| Validation perplexity over 20 training epochs | 94.6 → 2.07 |
| Model checkpoint size after dynamic int8 quantization | 4.550 MB → 1.557 MB |
| CPU forward-pass latency, fp32 → int8 | 3.221 ms → 2.855 ms |
| Greedy output in the quantization spot check | Identical on the tested prompt |

The earlier tokenizer comparison reported the following average subword pieces per word:

| Language | Dedicated tokenizer | Shared tokenizer |
|---|---:|---:|
| Hindi | 1.19 | 1.85 |
| Telugu | 1.29 | 2.44 |
| Malayalam | 1.30 | 2.90 |
| Kannada | 1.30 | 2.62 |

The shared tokenizer uses one vocabulary across all four scripts. In this experiment, that meant splitting words into more pieces than the dedicated tokenizers.

Reported decoding-diversity scores were 0.161 for greedy, 0.161 for beam search, 0.165 for top-k, and 0.157 for top-p. The evaluation script calls this metric “self-BLEU,” but implements a simplified pairwise n-gram overlap score. Lower scores indicate less overlap in the sampled outputs; they do not measure fluency or overall language quality.

These experiments use small, synthetic corpora. The results describe this setup and do not establish performance on everyday multilingual text.

## Configuration

[config.py](config.py) defines artifact paths, training settings, device selection, and logging. Common environment overrides include:

| Variable | Purpose |
|---|---|
| `ITD_DEVICE` | Device selection, such as `auto`, `cpu`, or `cuda` |
| `ITD_SEED` | Random seed |
| `ITD_LOG_LEVEL` | Logging level |
| `ITD_CORPUS` | Multilingual corpus path |
| `ITD_TOKENIZER` | SentencePiece model path |
| `ITD_LM_CKPT` / `ITD_CLS_CKPT` | Language-model and classifier checkpoint paths |
| `ITD_EPOCHS` / `ITD_BATCH_SIZE` / `ITD_LR` | Language-model training settings |
| `ITD_D_MODEL` / `ITD_N_HEADS` / `ITD_N_LAYERS` | Transformer dimensions |

For example, to train on CPU with a fixed seed:

```bash
ITD_DEVICE=cpu ITD_SEED=0 python -m model.train
```

## Project structure

| Path | What to look for |
|---|---|
| [data/](data/) | Synthetic corpus generation and language-tagged text |
| [tokenizer/](tokenizer/) | Handwritten BPE, SentencePiece training, and tokenizer files |
| [model/](model/) | TinyGPT and its training script |
| [decoding/](decoding/) | The four decoding algorithms |
| [classifier/](classifier/) | Sentiment data, BiLSTM training, and filtering demo |
| [custom_training/](custom_training/) | Custom loss, optimizer, and comparison benchmark |
| [quantization/](quantization/) | Dynamic int8 quantization and benchmarking |
| [eval/](eval/) | Perplexity, tokenizer-efficiency, and diversity measurements |
| [pipeline.py](pipeline.py) | Tokenizer, decoding, and filtering interfaces |
| [loaders.py](loaders.py) | Shared artifact loading and checkpoint validation |
| [api.py](api.py) | FastAPI service |
| [tests/](tests/) | Unit and integration tests |
| [docs/architecture/](docs/architecture/) | Archify source, preview, and interactive viewer |

## Limitations and next steps

The training data is generated from templates, so the model mainly demonstrates learning within a small, controlled vocabulary. Broader language quality needs evaluation on more varied text.

The component named `SafetyFilter` is a sentiment classifier trained on synthetic Hindi examples. It is not a general content-safety system, and its behavior should not be assumed to transfer to the other languages. The current pipeline returns the last candidate when every retry is rejected, so enabling the filter does not guarantee that the returned text passed it.

Useful next steps are to fix the API imports, make filter exhaustion explicit, evaluate on broader multilingual data, save reproducible benchmark reports, and add request batching and rate limiting for serving experiments.
