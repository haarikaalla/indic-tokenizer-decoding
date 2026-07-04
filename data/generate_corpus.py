"""
Generates an original, synthetic Hindi sentence corpus for tokenizer/LM training.

Why synthetic instead of scraped text?
- Avoids copyright issues entirely (all sentences are template-composed, not copied)
- Gives us control over vocabulary size and sentence diversity for a toy demo
- Real projects would swap this for a proper corpus (e.g. IndicCorp, Samanantar);
  the rest of the pipeline (tokenizer, LM, decoding, eval) doesn't change either way.
"""
import random
import itertools

random.seed(42)

subjects = ["राम", "सीता", "मोहन", "गीता", "अध्यापक", "छात्र", "किसान", "डॉक्टर",
            "बच्चे", "माँ", "पिता", "दादी", "दोस्त", "पड़ोसी", "वैज्ञानिक", "लेखक"]

verbs_present = ["जाता है", "आता है", "पढ़ता है", "खेलता है", "काम करता है",
                 "सोचता है", "मुस्कुराता है", "गाता है", "दौड़ता है", "सीखता है"]

verbs_present_f = ["जाती है", "आती है", "पढ़ती है", "खेलती है", "काम करती है",
                   "सोचती है", "मुस्कुराती है", "गाती है", "दौड़ती है", "सीखती है"]

female_subjects = {"सीता", "गीता", "माँ", "दादी", "बच्चे"}

places = ["स्कूल", "बाज़ार", "घर", "मंदिर", "पार्क", "खेत", "अस्पताल", "नदी किनारे",
          "पुस्तकालय", "गाँव", "शहर", "पहाड़ों पर"]

times = ["सुबह", "शाम को", "रोज़", "हर दिन", "आज", "कल", "सप्ताह में एक बार", "रात में"]

adjectives = ["बहुत खुश", "थोड़ा थका हुआ", "बहुत मेहनती", "बहुत समझदार",
              "बहुत दयालु", "बहुत जिज्ञासु"]

topics = ["किताबें", "विज्ञान", "संगीत", "प्रकृति", "गणित", "इतिहास", "कहानियाँ", "खेल"]

templates = [
    "{subj} {time} {place} {verb}।",
    "{subj} {adj} है और {topic} पसंद करता है।",
    "{subj} को {topic} में बहुत रुचि है।",
    "{time}, {subj} {place} {verb}।",
    "{subj} और उसका दोस्त {place} {verb}।",
]

def verb_for(subj, base_verb_idx):
    if subj in female_subjects:
        return verbs_present_f[base_verb_idx]
    return verbs_present[base_verb_idx]

def make_sentence():
    subj = random.choice(subjects)
    vi = random.randint(0, len(verbs_present) - 1)
    verb = verb_for(subj, vi)
    place = random.choice(places)
    time = random.choice(times)
    adj = random.choice(adjectives)
    topic = random.choice(topics)
    template = random.choice(templates)
    return template.format(subj=subj, verb=verb, place=place, time=time, adj=adj, topic=topic)

def main(n_sentences=1200, out_path="data/corpus_hi.txt"):
    seen = set()
    sentences = []
    attempts = 0
    while len(sentences) < n_sentences and attempts < n_sentences * 20:
        s = make_sentence()
        attempts += 1
        if s not in seen:
            seen.add(s)
            sentences.append(s)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences) + "\n")
    print(f"Wrote {len(sentences)} unique sentences to {out_path}")

if __name__ == "__main__":
    main()
