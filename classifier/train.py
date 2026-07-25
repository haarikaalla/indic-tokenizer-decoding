import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import random
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.classifier_model import TextClassifier

SP_MODEL = "tokenizer/multilingual_bpe.model"
DATA_PATH = "classifier/sentiment_data.tsv"
CKPT_PATH = "classifier/sentiment_classifier.pt"

def load_data(sp):
    examples = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            label, text = line.strip().split("\t", 1)
            ids = sp.encode(text, out_type=int)
            examples.append((torch.tensor(ids, dtype=torch.long), int(label)))
    random.seed(0)
    random.shuffle(examples)
    split = int(0.85 * len(examples))
    return examples[:split], examples[split:]

def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)

def make_batches(data, batch_size):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        yield collate(data[i:i + batch_size])

def evaluate(model, data, batch_size=32):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, lengths, y in make_batches(data, batch_size):
            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
    model.train()
    return correct / total

def main():
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    train_data, val_data = load_data(sp)
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    model = TextClassifier(vocab_size=sp.get_piece_size())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0.0
        n_batches = 0
        for x, lengths, y in make_batches(train_data, batch_size=32):
            logits = model(x, lengths)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        val_acc = evaluate(model, val_data)
        print(f"epoch {epoch+1:2d}/10 | train_loss {total_loss/n_batches:.3f} | val_acc {val_acc*100:.1f}%")

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nSaved classifier -> {CKPT_PATH}")

if __name__ == "__main__":
    main()
