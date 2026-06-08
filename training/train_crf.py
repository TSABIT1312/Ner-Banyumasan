"""
CRF training script for Banyumasan NER — v3 (hybrid pipeline).

Source data : data/processed/merged_hybrid.csv
              (5-label scheme: PER / LOC / ORG / TIME / MISC)
Changes vs v2:
  - Reads merged_hybrid.csv (hybrid auto+manual labels)
  - Filters sentences with ≤ 2 tokens (too short for CRF context window)
  - Stratified 80/10/10 split with fixed seed
  - Rebuilds LOC gazetteer and token-frequency table from training split
  - max_iterations = 400  (more features → needs more iterations)
  - all_possible_states = True  (added; previously only transitions)
  - Saves v2 model as backup before overwriting
  - Produces side-by-side v2 → v3 comparison in evaluation/report.txt

Usage:
    python training/train_crf.py
"""

import csv
import json
import random
import shutil
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import joblib
import sklearn_crfsuite
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR  = Path(__file__).parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR  = BASE_DIR / "evaluation"

RANDOM_SEED   = 42
MIN_SENT_LEN  = 3    # filter sentences shorter than this


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sentences(path: Path) -> list[list[tuple]]:
    sents = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sents[row["sentence_id"]].append((row["token"], row["label"]))
    return list(sents.values())


# ── Stratified split ──────────────────────────────────────────────────────────

def dominant_label(sent: list[tuple]) -> str:
    cnt = Counter(lbl for _, lbl in sent if lbl != "O")
    return cnt.most_common(1)[0][0] if cnt else "O"


def stratified_split(sents, train_r=0.8, val_r=0.1, seed=RANDOM_SEED):
    """Stratified 80/10/10 split by dominant entity label."""
    rng = random.Random(seed)
    buckets: dict = defaultdict(list)
    for s in sents:
        buckets[dominant_label(s)].append(s)

    train, val, test = [], [], []
    for lbl, group in buckets.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_r))
        n_val   = max(1, int(n * val_r))
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def save_split(sents: list[list[tuple]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence_id", "token", "label"])
        for sid, sent in enumerate(sents):
            for tok, lbl in sent:
                writer.writerow([sid, tok, lbl])


# ── Helpers ───────────────────────────────────────────────────────────────────

def flatten(seqs):
    return [item for seq in seqs for item in seq]


def label_distribution(y_seqs: list[list[str]]) -> str:
    counts = Counter(l for s in y_seqs for l in s)
    total  = sum(counts.values())
    lines  = []
    for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {lbl:15s}  {cnt:5d}  ({cnt/total*100:.1f}%)")
    return "\n".join(lines)


def entity_f1_summary(report_dict: dict) -> str:
    lines = [f"  {'Label':15s}  {'P':>6}  {'R':>6}  {'F1':>6}  {'Support':>8}"]
    lines.append("  " + "-" * 50)
    for lbl, vals in sorted(report_dict.items()):
        if isinstance(vals, dict) and lbl not in ("accuracy", "macro avg", "weighted avg"):
            lines.append(
                f"  {lbl:15s}  {vals['precision']:>6.3f}  {vals['recall']:>6.3f}"
                f"  {vals['f1-score']:>6.3f}  {vals['support']:>8}"
            )
    if "macro avg" in report_dict:
        v = report_dict["macro avg"]
        lines.append("  " + "-" * 50)
        lines.append(
            f"  {'macro avg':15s}  {v['precision']:>6.3f}  {v['recall']:>6.3f}"
            f"  {v['f1-score']:>6.3f}  {v['support']:>8}"
        )
    return "\n".join(lines)


def comparison_table(entity_labels, v_old, v_new, title_old="v2", title_new="v3") -> str:
    hdr  = f"  {'Label':12s}  {title_old:^26}  {title_new:^26}  {'ΔF1':>6}"
    sep  = "  " + "-" * 78
    lines = [hdr, sep]
    for lbl in entity_labels + ["macro avg"]:
        if lbl not in v_old or not isinstance(v_old[lbl], dict):
            continue
        d1 = v_old[lbl]
        d2 = v_new.get(lbl, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
        delta = d2["f1-score"] - d1["f1-score"]
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"  {lbl:12s}  "
            f"P={d1['precision']:.3f} R={d1['recall']:.3f} F={d1['f1-score']:.3f}  "
            f"P={d2['precision']:.3f} R={d2['recall']:.3f} F={d2['f1-score']:.3f}  "
            f"{sign}{delta:.3f}"
        )
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Banyumasan NER — CRF Training Pipeline  (v3 hybrid)")
    print("=" * 60)

    # ── Load merged hybrid data ────────────────────────────────
    print("\n[1] Loading merged_hybrid.csv...")
    all_sents = load_sentences(PROC_DIR / "merged_hybrid.csv")
    before = len(all_sents)
    all_sents = [s for s in all_sents if len(s) >= MIN_SENT_LEN]
    print(f"    Loaded {before} sentences; kept {len(all_sents)} after ≥{MIN_SENT_LEN}-token filter")

    # ── Stratified split ───────────────────────────────────────
    print("\n[2] Stratified 80/10/10 split (seed=42)...")
    train_sents, val_sents, test_sents = stratified_split(all_sents)
    print(f"    Train : {len(train_sents):4d}")
    print(f"    Val   : {len(val_sents):4d}")
    print(f"    Test  : {len(test_sents):4d}")

    save_split(train_sents, PROC_DIR / "train.csv")
    save_split(val_sents,   PROC_DIR / "val.csv")
    save_split(test_sents,  PROC_DIR / "test.csv")
    print("    Splits saved → data/processed/")

    # ── Build ancillary data from training split ───────────────
    print("\n[3] Building LOC gazetteer and token frequencies from train split...")
    import features as feat_mod

    loc_gaz  = set()
    tok_freq = Counter()
    for sent in train_sents:
        for tok, lbl in sent:
            tl = tok.lower()
            tok_freq[tl] += 1
            if lbl in ("B-LOC", "I-LOC"):
                loc_gaz.add(tl)

    feat_mod.LOC_GAZETTEER = loc_gaz
    feat_mod.TOKEN_FREQ    = dict(tok_freq)

    gaz_path  = MODEL_DIR / "loc_gazetteer.json"
    freq_path = MODEL_DIR / "token_freq.json"
    gaz_path.write_text(
        json.dumps(sorted(loc_gaz), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    freq_path.write_text(
        json.dumps(dict(tok_freq), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"    LOC gazetteer : {len(loc_gaz)} tokens  → {gaz_path.name}")
    print(f"    Token freq    : {len(tok_freq)} tokens  → {freq_path.name}")

    from features import sent_to_features, sent_to_labels

    # ── Backup v2 model and score it ──────────────────────────
    v2_path = MODEL_DIR / "crf_ner.joblib"
    v2_report: dict | None = None
    if v2_path.exists():
        bak = MODEL_DIR / "crf_ner_v2.joblib"
        shutil.copy2(v2_path, bak)
        print(f"\n[4] Backed up v2 model → {bak.name}")
        print("    Scoring v2 on new test split (empty gaz/freq for fair comparison)...")
        feat_mod.LOC_GAZETTEER = set()
        feat_mod.TOKEN_FREQ    = {}
        crf_v2   = joblib.load(bak)
        X_t_v2   = [sent_to_features(s) for s in test_sents]
        y_t_v2   = [sent_to_labels(s)   for s in test_sents]
        y_p_v2   = crf_v2.predict(X_t_v2)
        flat_t_v2 = flatten(y_t_v2)
        flat_p_v2 = flatten(y_p_v2)
        v2_labels  = sorted(set(flat_t_v2) - {"O"})
        v2_report  = classification_report(
            flat_t_v2, flat_p_v2,
            labels=v2_labels,
            zero_division=0,
            output_dict=True,
        )
        feat_mod.LOC_GAZETTEER = loc_gaz
        feat_mod.TOKEN_FREQ    = dict(tok_freq)
        print(entity_f1_summary(v2_report))
    else:
        print("\n[4] No v2 model found — skipping backup/comparison.")

    # ── Feature extraction ────────────────────────────────────
    print("\n[5] Extracting CRF features (v3)...")
    X_train = [sent_to_features(s) for s in train_sents]
    y_train = [sent_to_labels(s)   for s in train_sents]
    X_val   = [sent_to_features(s) for s in val_sents]
    y_val   = [sent_to_labels(s)   for s in val_sents]
    X_test  = [sent_to_features(s) for s in test_sents]
    y_test  = [sent_to_labels(s)   for s in test_sents]

    all_labels    = sorted(set(l for s in y_train + y_val + y_test for l in s))
    entity_labels = [l for l in all_labels if l != "O"]
    print(f"    Label set : {all_labels}")
    print("\n    Training label distribution:")
    print(label_distribution(y_train))

    # ── Train ─────────────────────────────────────────────────
    print("\n[6] Training CRF v3 (lbfgs, c1=0.025, c2=0.1, max_iter=400)...")
    t0  = time.time()
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.025,
        c2=0.1,
        max_iterations=400,
        all_possible_transitions=True,
        all_possible_states=True,
    )
    crf.fit(X_train, y_train)
    print(f"    Done in {time.time() - t0:.1f}s")

    # ── Validation ────────────────────────────────────────────
    print("\n[7] Validation:")
    y_val_pred = crf.predict(X_val)
    val_flat_t = flatten(y_val);  val_flat_p = flatten(y_val_pred)
    val_report = classification_report(
        val_flat_t, val_flat_p,
        labels=entity_labels,
        zero_division=0,
        output_dict=True,
    )
    val_report_str = classification_report(
        val_flat_t, val_flat_p,
        labels=entity_labels,
        zero_division=0,
    )
    print(entity_f1_summary(val_report))

    # ── Test ──────────────────────────────────────────────────
    print("\n[8] Test (v3):")
    y_test_pred  = crf.predict(X_test)
    flat_true_v3 = flatten(y_test);  flat_pred_v3 = flatten(y_test_pred)
    v3_report = classification_report(
        flat_true_v3, flat_pred_v3,
        labels=entity_labels,
        zero_division=0,
        output_dict=True,
    )
    v3_report_str = classification_report(
        flat_true_v3, flat_pred_v3,
        labels=entity_labels,
        zero_division=0,
    )
    print(entity_f1_summary(v3_report))

    # ── Comparison ────────────────────────────────────────────
    cmp_table = "(no v2 baseline)"
    if v2_report is not None:
        all_cmp_labels = sorted(set(entity_labels + [l for l in v2_report
                                                      if isinstance(v2_report[l], dict)
                                                      and l not in ("macro avg","weighted avg","accuracy")]))
        print("\n[9] v2 → v3 comparison (test set):")
        cmp_table = comparison_table(all_cmp_labels, v2_report, v3_report)
        print(cmp_table)

    # ── Top transitions ────────────────────────────────────────
    trans = sorted(crf.transition_features_.items(), key=lambda x: -x[1])
    print("\n[10] Top transitions (v3):")
    for (frm, to), weight in trans[:12]:
        print(f"    {frm:15s} → {to:15s}  {weight:+.3f}")

    # ── Save ─────────────────────────────────────────────────
    print("\n[11] Saving artifacts...")
    joblib.dump(crf, MODEL_DIR / "crf_ner.joblib")
    print("    Model  → models/crf_ner.joblib")

    (MODEL_DIR / "label_encoder.json").write_text(
        json.dumps({"labels": all_labels, "entity_labels": entity_labels}, indent=2),
        encoding="utf-8",
    )
    print("    Labels → models/label_encoder.json")

    # ── Write report ──────────────────────────────────────────
    report_path = EVAL_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Banyumasan NER — Full Evaluation Report  (v3 hybrid pipeline)\n")
        f.write("=" * 60 + "\n\n")
        f.write("=== Per-label metrics (test set) ===\n")
        f.write(v3_report_str + "\n")
        f.write("=== Validation set ===\n")
        f.write(val_report_str + "\n")
        f.write("=== v2 → v3 comparison (test set) ===\n")
        f.write(cmp_table + "\n\n")
        f.write("=== LOC gazetteer ===\n")
        f.write(", ".join(sorted(loc_gaz)) + "\n\n")
        f.write("=== Top transitions (v3) ===\n")
        for (frm, to), weight in trans[:20]:
            f.write(f"  {frm:15s} → {to:15s}  {weight:+.3f}\n")
    print(f"    Report → evaluation/report.txt")

    print("\n" + "=" * 60)
    print("  Training complete.  5-label model saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
