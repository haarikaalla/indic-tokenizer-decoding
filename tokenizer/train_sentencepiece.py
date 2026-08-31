"""
Trains a subword (BPE) tokenizer on the Hindi corpus using SentencePiece.

This is the "use the production tool correctly" half of the tokenizer story.
See bpe_from_scratch.py for the "implement the algorithm myself" half.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import sentencepiece as spm

if __package__ in (None, ""):  # `python tokenizer/train_sentencepiece.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

logger = logging.getLogger("tokenizer")

# One multilingual tokenizer trained on all languages together (mirrors mT5/mBART
# approach: a single shared subword vocabulary across languages, so the model can
# share representations for structurally similar words/scripts where possible).
#
# Per-language tokenizers are trained separately, so we can measure whether a shared
# multilingual vocabulary is more or less efficient per language than a dedicated one.
LANG_TAG_SYMBOLS = [f"<{code}>" for code in config.LANGUAGES]


def train_spm(input_path: Path, model_prefix: Path, vocab_size: int) -> None:
    """Train one SentencePiece BPE model, failing early if the corpus is missing."""
    corpus = config.require_file(input_path, "Generate the corpora first: python data/generate_corpus.py")
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,       # must be 1.0 for non-Latin scripts
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        user_defined_symbols=LANG_TAG_SYMBOLS,  # language tags as atomic tokens
    )
    logger.info("Trained tokenizer -> %s.model / .vocab", model_prefix)


def train_all() -> None:
    """Train the shared multilingual tokenizer plus one dedicated tokenizer per language."""
    train_spm(config.MULTILINGUAL_CORPUS, config.TOKENIZER_PREFIX, config.TOKENIZER_VOCAB_SIZE)
    for lang, path in config.LANG_CORPUS.items():
        train_spm(path, config.TOKENIZER_DIR / f"{lang}_bpe", config.PER_LANG_VOCAB_SIZE)


def _avg_pieces_per_word(sp: spm.SentencePieceProcessor, lines: list[str]) -> float:
    words, pieces = 0, 0
    for line in lines:
        words += len(line.split())
        pieces += len(sp.encode(line, out_type=int))
    return pieces / words if words else 0.0


def compare_efficiency(sample: int = 200) -> dict[str, tuple[float, float]]:
    """For each language, measure subwords-per-word using (a) its own dedicated
    tokenizer vs (b) the shared multilingual tokenizer. Returns {lang: (own, shared)}."""
    multi_sp = spm.SentencePieceProcessor(
        model_file=str(config.require_file(f"{config.TOKENIZER_PREFIX}.model"))
    )
    logger.info("Tokenizer efficiency: dedicated vs. shared multilingual")

    results: dict[str, tuple[float, float]] = {}
    for lang, path in config.LANG_CORPUS.items():
        own_model = config.require_file(config.TOKENIZER_DIR / f"{lang}_bpe.model")
        own_sp = spm.SentencePieceProcessor(model_file=str(own_model))
        with config.require_file(path).open(encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()][:sample]

        results[lang] = (_avg_pieces_per_word(own_sp, lines), _avg_pieces_per_word(multi_sp, lines))
        logger.info(
            "[%s] dedicated tokenizer: %.2f pieces/word  |  shared multilingual tokenizer: %.2f pieces/word",
            lang, results[lang][0], results[lang][1],
        )
    return results


def demo() -> None:
    """Show how the shared tokenizer segments one sentence per language."""
    sp = spm.SentencePieceProcessor(
        model_file=str(config.require_file(f"{config.TOKENIZER_PREFIX}.model"))
    )
    samples = [
        ("<hi>", "राम स्कूल जाता है।"),
        ("<te>", "విద్యార్థి పాఠశాలకు వెళ్తాడు."),
        ("<ml>", "വിദ്യാർത്ഥി സ്കൂളിൽ പോകുന്നു."),
        ("<kn>", "ವಿದ್ಯಾರ್ಥಿ ಶಾಲೆಗೆ ಹೋಗುತ್ತಾನೆ."),
    ]
    logger.info("Shared multilingual tokenizer demo")
    for tag, sentence in samples:
        text = f"{tag} {sentence}"
        logger.info("Input  : %s", text)
        logger.info("Pieces : %s", sp.encode(text, out_type=str))


def main() -> None:
    config.setup_logging()
    train_all()
    compare_efficiency()
    demo()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
