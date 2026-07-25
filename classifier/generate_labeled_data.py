"""
Generates a small labeled sentiment dataset (positive / negative) in Hindi,
reusing the same synthetic-template approach as the main corpus (original
content, no copyright risk).

This classifier later gets wired into generate.py as an output filter --
directly demonstrating "safety-controlled text composition" from the JD:
the generation pipeline isn't just producing raw text, it's gating it.
"""
import random
random.seed(7)

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

def make_examples(adjectives, label, n):
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
    return out

def main(n_per_class=400, out_path="classifier/sentiment_data.tsv"):
    pos = make_examples(positive_adjectives, 1, n_per_class)
    neg = make_examples(negative_adjectives, 0, n_per_class)
    all_data = pos + neg
    random.shuffle(all_data)
    with open(out_path, "w", encoding="utf-8") as f:
        for text, label in all_data:
            f.write(f"{label}\t{text}\n")
    print(f"Wrote {len(all_data)} labeled examples ({len(pos)} positive, {len(neg)} negative) -> {out_path}")

if __name__ == "__main__":
    main()
