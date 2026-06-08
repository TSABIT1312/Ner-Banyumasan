"""
Standalone evaluation script — runs after train_crf.py.

Produces:
  - Per-label precision / recall / F1
  - Confusion matrix (entity classes only)
  - Per-sentence error analysis (sentences with mispredictions)
  - evaluation/report.txt (overwritten)

Usage:
    python evaluation/evaluate.py
"""

import csv
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
PROC_DIR  = BASE_DIR / "data" / "processed"
EVAL_DIR  = Path(__file__).parent

sys.path.insert(0, str(BASE_DIR / "training"))
from features import sent_to_features, sent_to_labels


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sentences(path: Path) -> list[list[tuple]]:
    sents: dict = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sents[row["sentence_id"]].append((row["token"], row["label"]))
    return list(sents.values())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Banyumasan NER — Evaluation Report")
    print("=" * 60)

    crf = joblib.load(MODEL_DIR / "crf_ner.joblib")
    with open(MODEL_DIR / "label_encoder.json", encoding="utf-8") as f:
        info = json.load(f)
    entity_labels = info["entity_labels"]

    test_sents = load_sentences(PROC_DIR / "test.csv")
    X_test     = [sent_to_features(s) for s in test_sents]
    y_true     = [sent_to_labels(s)   for s in test_sents]
    y_pred     = crf.predict(X_test)

    flat_true = [l for s in y_true for l in s]
    flat_pred = [l for s in y_pred for l in s]

    # ── Classification report ─────────────────────────────────
    report_str = classification_report(
        flat_true, flat_pred,
        labels=entity_labels,
        zero_division=0,
    )
    print("\n── Per-label metrics (test set) ──")
    print(report_str)

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(flat_true, flat_pred, labels=entity_labels)
    print("── Confusion matrix (entity labels only) ──")
    col_w = 14
    header = f"{'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in entity_labels)
    print(header)
    for i, row_lbl in enumerate(entity_labels):
        row_str = f"{row_lbl:>{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(entity_labels)))
        print(row_str)

    # ── Error analysis: sentences with at least one wrong entity ─
    print("\n── Error analysis (first 10 sentences with entity errors) ──")
    shown = 0
    for sent, true_seq, pred_seq in zip(test_sents, y_true, y_pred):
        has_error = any(
            t != p and (t != "O" or p != "O")
            for t, p in zip(true_seq, pred_seq)
        )
        if has_error and shown < 10:
            tokens = [tok for tok, _ in sent]
            print(f"\n  Sentence: {' '.join(tokens)}")
            for tok, true_lbl, pred_lbl in zip(tokens, true_seq, pred_seq):
                if true_lbl != pred_lbl:
                    print(f"    '{tok}':  TRUE={true_lbl}  PRED={pred_lbl}")
            shown += 1

    # ── Per-entity-type accuracy ──────────────────────────────
    print("\n── Entity-type level accuracy ──")
    type_correct: dict = defaultdict(int)
    type_total:   dict = defaultdict(int)
    for t, p in zip(flat_true, flat_pred):
        if t != "O":
            etype = t.split("-", 1)[1]
            type_total[etype] += 1
            if t == p:
                type_correct[etype] += 1
    for etype in sorted(type_total):
        acc = type_correct[etype] / type_total[etype] * 100
        print(f"  {etype:8s}  {type_correct[etype]:3d}/{type_total[etype]:3d}  ({acc:.1f}% token-level accuracy)")

    # ── Save report ───────────────────────────────────────────
    EVAL_DIR.mkdir(exist_ok=True)
    with open(EVAL_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write("Banyumasan NER — Full Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("=== Per-label metrics (test set) ===\n")
        f.write(report_str + "\n")
        f.write("=== Confusion matrix ===\n")
        f.write(header + "\n")
        for i, row_lbl in enumerate(entity_labels):
            f.write(f"{row_lbl:>{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(entity_labels))) + "\n")
        f.write("\n=== Entity-type token accuracy ===\n")
        for etype in sorted(type_total):
            acc = type_correct[etype] / type_total[etype] * 100
            f.write(f"  {etype:8s}  {type_correct[etype]:3d}/{type_total[etype]:3d}  ({acc:.1f}%)\n")

    print(f"\nReport saved → evaluation/report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
