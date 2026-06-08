"""
Hybrid label merger for Banyumasan NER.

Strategy (Option C):
  Priority 1 — auto_label.py tags (PER/LOC/ORG/TIME): always trusted.
  Priority 2 — Manual annotations fill the rest after noise filtering:
    MISC  : kept as-is (objects/items are legitimate MISC)
    LOC   : kept unless token is in LOC_NOISE (teacher, internet, etc.)
    ORG   : kept as-is
    PER   : kept only if token is NOT a pronoun/common noun AND
            the B- token is capitalized

Result: 5-label BIO scheme  PER / LOC / ORG / TIME / MISC
Output: data/processed/merged_hybrid.csv
"""

import csv
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from auto_label import tag_sentence

BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "data" / "processed"

# ── Noise filters ─────────────────────────────────────────────────────────────

# Banyumasan/Javanese pronouns and common-noun person referents — should NOT be PER
PER_NOISE = {
    "inyong", "nyong", "inyongrika",      # 1st-person pronouns (I/me)
    "kula", "aku", "ingsun",              # 1st-person pronouns (formal/archaic)
    "awake", "awakedhewe",                # reflexive (himself/herself/ourselves)
    "bocah", "bocah-bocah",               # child, children
    "wong", "uwong",                      # person (generic)
    "anak",                               # child
    "biyung", "biyunge", "emak",          # mother
    "wali",                               # guardian / representative
    "guru",                               # teacher (also annotated as LOC and MISC)
    "pak", "bu", "mas", "mbak", "kang", "yu",  # titles alone (auto handles title+name)
}

# LOC tokens that are clearly mis-annotated in manual data
LOC_NOISE = {
    "internet",   # not a physical location
    "guru",       # teacher
    "kepala",     # head / principal (person role)
    "kolom",      # column / field (text element)
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw_file(path: Path, encoding: str) -> dict[str, list[tuple[str, str]]]:
    """Return {sentence_id: [(token, label), ...]}."""
    sents = defaultdict(list)
    with open(path, encoding=encoding) as f:
        for row in csv.DictReader(f):
            sents[row["sentence_id"]].append((row["token"], row["label"]))
    return dict(sents)


# ── BIO fixer ─────────────────────────────────────────────────────────────────

def fix_bio(labels: list[str]) -> list[str]:
    """
    Two-pass BIO repair (mirrors preprocess.py behaviour on raw B-only data).

    Pass 1: consecutive B-X B-X of the same type → B-X I-X
            (raw files are B-only; this reconstructs multi-token spans)
    Pass 2: any I-X not following B-X/I-X of the same type → B-X
            (repairs orphaned I- tags)
    """
    fixed = list(labels)
    # Pass 1
    for i in range(1, len(fixed)):
        if (fixed[i].startswith("B-")
                and fixed[i - 1].startswith("B-")
                and fixed[i][2:] == fixed[i - 1][2:]):
            fixed[i] = "I-" + fixed[i][2:]
    # Pass 2
    for i, lbl in enumerate(fixed):
        if lbl.startswith("I-"):
            etype = lbl[2:]
            prev  = fixed[i - 1] if i > 0 else "O"
            if not (prev == f"B-{etype}" or prev == f"I-{etype}"):
                fixed[i] = f"B-{etype}"
    return fixed


# ── Hybrid merge ──────────────────────────────────────────────────────────────

def merge(tokens: list[str], auto_labels: list[str], manual_labels: list[str]) -> list[str]:
    merged = []
    for tok, auto, manual in zip(tokens, auto_labels, manual_labels):
        tl = tok.lower()

        # Rule 1: auto_label non-O always wins
        if auto != "O":
            merged.append(auto)
            continue

        # Rule 2: auto=O — consider manual label
        if manual == "O":
            merged.append("O")
            continue

        # Strip BIO prefix to get entity type
        m_type = manual[2:]   # e.g. "PER", "LOC"

        if m_type == "MISC":
            merged.append(manual)

        elif m_type == "LOC":
            if tl in LOC_NOISE:
                merged.append("O")
            else:
                merged.append(manual)

        elif m_type == "ORG":
            merged.append(manual)

        elif m_type == "PER":
            is_b_tag   = manual.startswith("B-")
            is_cap     = tok[0].isupper() if tok else False
            is_noisy   = tl in PER_NOISE
            if is_noisy:
                merged.append("O")
            elif is_b_tag and not is_cap:
                # B-PER for a lowercase token — almost always a common noun
                merged.append("O")
            else:
                merged.append(manual)

        else:
            # Unknown type — drop
            merged.append("O")

    return fix_bio(merged)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Banyumasan NER — Hybrid Label Merger")
    print("=" * 60)

    raw_files = [
        (RAW_DIR / "project15_clean_training.csv", "utf-8-sig", 0),
        (RAW_DIR / "project16_clean_training.csv", "utf-8-sig", 1),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "merged_hybrid.csv"

    counts      = Counter()
    bio_errors  = 0
    n_sent      = 0
    n_tok       = 0
    seen_texts  = set()     # for dedup

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["sentence_id", "token", "label"])

        for path, enc, doc_id in raw_files:
            raw = load_raw_file(path, enc)
            print(f"\n  [{path.name}] — {len(raw)} sentences")

            for orig_sid in sorted(raw, key=lambda x: int(x)):
                sent = raw[orig_sid]
                if not sent:
                    continue

                tokens = [tok for tok, _ in sent]
                manual = [lbl for _, lbl in sent]

                # Dedup by whitespace-joined token sequence
                text_key = " ".join(t.lower() for t in tokens)
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                # auto_label tagger
                auto = tag_sentence(tokens)

                # Hybrid merge
                merged = merge(tokens, auto, manual)

                # BIO validation check
                prev = "O"
                for lbl in merged:
                    if lbl.startswith("I-"):
                        etype = lbl[2:]
                        if prev not in (f"B-{etype}", f"I-{etype}"):
                            bio_errors += 1
                    prev = lbl

                # Write
                for tok, lbl in zip(tokens, merged):
                    writer.writerow([n_sent, tok, lbl])
                    n_tok += 1
                    if lbl.startswith("B-"):
                        counts[lbl[2:]] += 1

                n_sent += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n  Output  → {out_path}")
    print(f"\n  Sentences  : {n_sent}")
    print(f"  Tokens     : {n_tok}")
    print(f"  BIO errors : {bio_errors}")

    # Label distribution
    all_labels = []
    with open(out_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_labels.append(row["label"])
    lbl_cnt = Counter(all_labels)
    n_total = len(all_labels)

    print(f"\n  Label distribution:")
    for lbl, cnt in sorted(lbl_cnt.items(), key=lambda x: -x[1]):
        print(f"    {lbl:12s} {cnt:6d}  ({cnt/n_total*100:.2f}%)")

    print(f"\n  Entity spans (B- counts):")
    for etype in ["PER", "LOC", "ORG", "TIME", "MISC"]:
        b = lbl_cnt.get(f"B-{etype}", 0)
        i = lbl_cnt.get(f"I-{etype}", 0)
        total = b + i
        print(f"    {etype:6s}: {b} B-spans, {i} I-tokens  ({total} total)")

    print("\n" + "=" * 60)
    print("  Merge complete. Run training/train_crf.py to retrain.")
    print("=" * 60)


if __name__ == "__main__":
    main()
