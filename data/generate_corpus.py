"""
Generates original, synthetic sentence corpora in Hindi, Telugu, and Malayalam.

Same rationale as before: template-composed, not scraped, so there's zero copyright
risk and full control over vocabulary/size. This version adds two more Indic
languages and produces:
  - one corpus file per language (for per-language tokenizer comparison)
  - one COMBINED, language-tagged corpus (for training a single multilingual model)

Language tags (<hi>, <te>, <ml>) prepended to each line follow the same pattern
used in real multilingual models like mBART/mT5 to condition generation on language.
"""
import random

random.seed(42)

LANGUAGES = {
    "hi": {
        "subjects": ["राम", "सीता", "मोहन", "गीता", "अध्यापक", "छात्र", "किसान", "डॉक्टर",
                     "बच्चे", "माँ", "पिता", "दादी", "दोस्त", "पड़ोसी", "वैज्ञानिक", "लेखक"],
        "verbs": ["जाता है", "आता है", "पढ़ता है", "खेलता है", "काम करता है",
                  "सोचता है", "मुस्कुराता है", "गाता है", "दौड़ता है", "सीखता है"],
        "places": ["स्कूल", "बाज़ार", "घर", "मंदिर", "पार्क", "खेत", "अस्पताल",
                   "नदी किनारे", "पुस्तकालय", "गाँव", "शहर", "पहाड़ों पर"],
        "times": ["सुबह", "शाम को", "रोज़", "हर दिन", "आज", "कल", "सप्ताह में एक बार", "रात में"],
        "adjectives": ["बहुत खुश", "थोड़ा थका हुआ", "बहुत मेहनती", "बहुत समझदार",
                       "बहुत दयालु", "बहुत जिज्ञासु"],
        "topics": ["किताबें", "विज्ञान", "संगीत", "प्रकृति", "गणित", "इतिहास", "कहानियाँ", "खेल"],
        "templates": [
            "{subj} {time} {place} {verb}।",
            "{subj} {adj} है और {topic} पसंद करता है।",
            "{subj} को {topic} में बहुत रुचि है।",
            "{time}, {subj} {place} {verb}।",
        ],
    },
    "te": {
        "subjects": ["రాముడు", "సీత", "మోహన్", "గీత", "ఉపాధ్యాయుడు", "విద్యార్థి", "రైతు", "డాక్టర్",
                     "పిల్లలు", "అమ్మ", "నాన్న", "నానమ్మ", "స్నేహితుడు", "పొరుగువాడు", "శాస్త్రవేత్త", "రచయిత"],
        "verbs": ["వెళ్తాడు", "వస్తాడు", "చదువుతాడు", "ఆడుతాడు", "పని చేస్తాడు",
                  "ఆలోచిస్తాడు", "నవ్వుతాడు", "పాడతాడు", "పరిగెత్తుతాడు", "నేర్చుకుంటాడు"],
        "places": ["పాఠశాల", "బజార్", "ఇల్లు", "గుడి", "పార్క్", "పొలం", "ఆసుపత్రి",
                   "నది ఒడ్డున", "గ్రంథాలయం", "గ్రామం", "పట్టణం", "కొండలపై"],
        "times": ["ఉదయం", "సాయంత్రం", "ప్రతిరోజు", "ఈరోజు", "రేపు", "వారానికి ఒకసారి", "రాత్రి"],
        "adjectives": ["చాలా సంతోషంగా", "కొంచెం అలసిపోయి", "చాలా కష్టపడేవాడు", "చాలా తెలివైన",
                       "చాలా దయగల", "చాలా ఆసక్తిగల"],
        "topics": ["పుస్తకాలు", "విజ్ఞానశాస్త్రం", "సంగీతం", "ప్రకృతి", "గణితం", "చరిత్ర", "కథలు", "ఆటలు"],
        "templates": [
            "{subj} {time} {place} {verb}.",
            "{subj} {adj}, మరియు {topic} ఇష్టపడతాడు.",
            "{subj} కి {topic} అంటే చాలా ఆసక్తి.",
            "{time}, {subj} {place} {verb}.",
        ],
    },
    "ml": {
        "subjects": ["രാമൻ", "സീത", "മോഹൻ", "ഗീത", "അധ്യാപകൻ", "വിദ്യാർത്ഥി", "കർഷകൻ", "ഡോക്ടർ",
                     "കുട്ടികൾ", "അമ്മ", "അച്ഛൻ", "അമ്മൂമ്മ", "സുഹൃത്ത്", "അയൽക്കാരൻ", "ശാസ്ത്രജ്ഞൻ", "എഴുത്തുകാരൻ"],
        "verbs": ["പോകുന്നു", "വരുന്നു", "വായിക്കുന്നു", "കളിക്കുന്നു", "ജോലി ചെയ്യുന്നു",
                  "ചിന്തിക്കുന്നു", "ചിരിക്കുന്നു", "പാടുന്നു", "ഓടുന്നു", "പഠിക്കുന്നു"],
        "places": ["സ്കൂൾ", "ചന്ത", "വീട്", "അമ്പലം", "പാർക്ക്", "പാടം", "ആശുപത്രി",
                   "നദിക്കരയിൽ", "ലൈബ്രറി", "ഗ്രാമം", "പട്ടണം", "മലകളിൽ"],
        "times": ["രാവിലെ", "വൈകുന്നേരം", "എല്ലാ ദിവസവും", "ഇന്ന്", "നാളെ", "ആഴ്ചയിൽ ഒരിക്കൽ", "രാത്രിയിൽ"],
        "adjectives": ["വളരെ സന്തോഷമുള്ള", "അല്പം ക്ഷീണിച്ച", "വളരെ കഠിനാധ്വാനിയായ", "വളരെ ബുദ്ധിമാനായ",
                       "വളരെ ദയയുള്ള", "വളരെ ജിജ്ഞാസയുള്ള"],
        "topics": ["പുസ്തകങ്ങൾ", "ശാസ്ത്രം", "സംഗീതം", "പ്രകൃതി", "ഗണിതം", "ചരിത്രം", "കഥകൾ", "കളികൾ"],
        "templates": [
            "{subj} {time} {place} {verb}.",
            "{subj} {adj}, {topic} ഇഷ്ടപ്പെടുന്നു.",
            "{subj}ക്ക് {topic} ൽ വലിയ താല്പര്യം ഉണ്ട്.",
            "{time}, {subj} {place} {verb}.",
        ],
    },
}


def make_sentence(lang_cfg):
    subj = random.choice(lang_cfg["subjects"])
    verb = random.choice(lang_cfg["verbs"])
    place = random.choice(lang_cfg["places"])
    time = random.choice(lang_cfg["times"])
    adj = random.choice(lang_cfg["adjectives"])
    topic = random.choice(lang_cfg["topics"])
    template = random.choice(lang_cfg["templates"])
    return template.format(subj=subj, verb=verb, place=place, time=time, adj=adj, topic=topic)


def generate_language(lang_code, n_sentences=1200):
    cfg = LANGUAGES[lang_code]
    seen, sentences, attempts = set(), [], 0
    while len(sentences) < n_sentences and attempts < n_sentences * 20:
        s = make_sentence(cfg)
        attempts += 1
        if s not in seen:
            seen.add(s)
            sentences.append(s)
    return sentences


def main(n_per_lang=1200):
    all_tagged_lines = []
    for lang_code in LANGUAGES:
        sentences = generate_language(lang_code, n_per_lang)
        out_path = f"data/corpus_{lang_code}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences) + "\n")
        print(f"[{lang_code}] wrote {len(sentences)} unique sentences -> {out_path}")

        tagged = [f"<{lang_code}> {s}" for s in sentences]
        all_tagged_lines.extend(tagged)

    random.shuffle(all_tagged_lines)
    with open("data/corpus_multilingual.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_tagged_lines) + "\n")
    print(f"\n[combined] wrote {len(all_tagged_lines)} tagged sentences -> data/corpus_multilingual.txt")


if __name__ == "__main__":
    main()
