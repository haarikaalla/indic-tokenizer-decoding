"""
Generates a small labeled sentiment dataset (positive / negative) in Hindi,
reusing the same synthetic-template approach as the main corpus (original
content, no copyright risk).

This classifier later gets wired into generate.py as an output filter --
directly demonstrating "safety-controlled text composition" from the JD:
the generation pipeline isn't just producing raw text, it's gating it.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

if __package__ in (None, ""):  # `python classifier/generate_labeled_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

logger = logging.getLogger("labeled_data")

subjects = ["राम", "सीता", "मोहन", "गीता", "अध्यापक", "छात्र", "किसान", "डॉक्टर",
            "बच्चे", "माँ", "पिता", "दोस्त"]

positive_adjectives = ["बहुत खुश", "बहुत उत्साहित", "बहुत संतुष्ट", "बहुत आशावादी",
                        "बहुत मेहनती", "बहुत दयालु"]
negative_adjectives = ["बहुत उदास", "बहुत गुस्से में", "बहुत निराश", "बहुत चिंतित",
                        "बहुत आलसी", "बहुत क्रोधित"]

templates = [
    "{subj} आज {adj} है।",
    "{subj} {adj} महसूस कर रहा है।",
    "इस समय {subj} {adj} है।",
]

def make_examples(adjectives: list[str], label: int, n: int) -> list[tuple[str, int]]:
    """Compose up to `n` unique labeled sentences from the templates."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    seen, out = set(), []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        subj = random.choice(subjects)
        adj = random.choice(adjectives)
        template = random.choice(templates)
        s = template.format(subj=subj, adj=adj)
        if s not in seen:
            seen.add(s)
            out.append((s, label))
    if len(out) < n:
        logger.warning("Templates exhausted for label %d: produced %d/%d examples.", label, len(out), n)
    return out


def main(n_per_class: int = 400, out_path=None) -> None:
    config.setup_logging()
    random.seed(7)

    target = Path(out_path) if out_path else config.SENTIMENT_DATA
    target.parent.mkdir(parents=True, exist_ok=True)

    pos = make_examples(positive_adjectives, 1, n_per_class)
    neg = make_examples(negative_adjectives, 0, n_per_class)
    all_data = pos + neg
    random.shuffle(all_data)

    target.write_text(
        "".join(f"{label}\t{text}\n" for text, label in all_data),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %d labeled examples (%d positive, %d negative) -> %s",
        len(all_data), len(pos), len(neg), target,
    )


if __name__ == "__main__":
    main()
