"""
Preprocessing pipeline for Banyumasan NER datasets.

Steps executed in order:
  1. Load project15 and project16 CSVs (utf-8-sig strips BOM)
  2. Normalize token whitespace
  3. Deduplicate identical sentences within each file
  4. Re-number sentence IDs to eliminate collision
     (p15 → 1..N,  p16 → N+1..M)
  5. Convert flat B-only scheme to proper BIO:
     consecutive B-X B-X of the same type → B-X I-X
  6. Save: data/processed/merged.csv, merged_bio.csv
  7. Stratified 80/10/10 split at sentence level
     (stratified by dominant entity class per sentence)
  8. Save: data/processed/train.csv, val.csv, test.csv
"""

import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

SEED     = 42
BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            token = row["token"].strip()
            label = row["label"].strip()
            if token:
                rows.append({
                    "sentence_id": row["sentence_id"].strip(),
                    "token":       token,
                    "label":       label,
                })
    return rows


def group_by_sentence(rows: list[dict]) -> dict:
    sents = defaultdict(list)
    for r in rows:
        sents[r["sentence_id"]].append((r["token"], r["label"]))
    return dict(sents)


def write_csv(sents: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "token", "label"])
        for sid, toks in sents.items():
            for tok, lbl in toks:
                w.writerow([sid, tok, lbl])


# ── Cleaning ──────────────────────────────────────────────────────────────────

def dedup(sents: dict) -> dict:
    seen, result = set(), {}
    for sid, toks in sents.items():
        key = " ".join(t for t, _ in toks)
        if key not in seen:
            seen.add(key)
            result[sid] = toks
    return result


def renumber(sent_lists: list[list], start: int = 1) -> dict:
    return {str(start + i): tl for i, tl in enumerate(sent_lists)}


def fix_bio(tokens_labels: list[tuple]) -> list[tuple]:
    """
    Convert flat B-only annotation to proper BIO.
    Rule: if token[i] is B-X and token[i-1] is B-X or I-X of the same type
          → relabel token[i] as I-X.
    Different entity types keep their B- prefix (they are separate spans).
    """
    result = []
    for i, (tok, lbl) in enumerate(tokens_labels):
        if lbl.startswith("B-") and result:
            prev_lbl = result[-1][1]
            if prev_lbl.startswith(("B-", "I-")):
                prev_type = prev_lbl.split("-", 1)[1]
                curr_type = lbl.split("-", 1)[1]
                if prev_type == curr_type:
                    lbl = "I-" + curr_type
        result.append((tok, lbl))
    return result


# ── Splitting ─────────────────────────────────────────────────────────────────

def dominant_label(toks: list[tuple]) -> str:
    counts = Counter(lbl.split("-", 1)[1] for _, lbl in toks if lbl != "O")
    return counts.most_common(1)[0][0] if counts else "O"


def stratified_split(
    sents: dict,
    train_r: float = 0.8,
    val_r:   float = 0.1,
    seed:    int   = SEED,
) -> tuple[dict, dict, dict]:
    groups: dict[str, list] = defaultdict(list)
    for sid, toks in sents.items():
        groups[dominant_label(toks)].append(sid)

    rng = random.Random(seed)
    train_ids, val_ids, test_ids = [], [], []

    for grp, ids in groups.items():
        rng.shuffle(ids)
        n       = len(ids)
        n_val   = max(1, round(n * val_r))
        n_test  = max(1, round(n * (1 - train_r - val_r)))
        n_train = n - n_val - n_test
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    def subset(ids):
        return {sid: sents[sid] for sid in ids}

    return subset(train_ids), subset(val_ids), subset(test_ids)


# ── Stats printer ─────────────────────────────────────────────────────────────

def print_stats(label: str, sents: dict) -> None:
    all_toks = [(t, l) for toks in sents.values() for t, l in toks]
    lc = Counter(l for _, l in all_toks)
    total = len(all_toks)
    print(f"\n  [{label}]  {len(sents)} sentences, {total} tokens")
    for lbl, cnt in sorted(lc.items(), key=lambda x: -x[1]):
        print(f"    {lbl:12s}  {cnt:5d}  ({cnt/total*100:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Banyumasan NER — Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading raw files...")
    rows15 = load_csv(RAW_DIR / "project15_clean_training.csv")
    rows16 = load_csv(RAW_DIR / "project16_clean_training.csv")
    s15    = group_by_sentence(rows15)
    s16    = group_by_sentence(rows16)
    print(f"    p15: {len(s15)} sentences, {len(rows15)} tokens")
    print(f"    p16: {len(s16)} sentences, {len(rows16)} tokens")

    # 2. Deduplicate
    print("\n[2] Deduplicating...")
    s15 = dedup(s15);  s16 = dedup(s16)
    print(f"    After dedup — p15: {len(s15)}, p16: {len(s16)}")

    # 3. Re-number and merge
    print("\n[3] Merging with collision-free sentence IDs...")
    merged_raw = renumber(list(s15.values()), start=1)
    offset     = len(s15) + 1
    for i, tl in enumerate(s16.values()):
        merged_raw[str(offset + i)] = tl
    total_sents = len(merged_raw)
    total_toks  = sum(len(v) for v in merged_raw.values())
    print(f"    Merged: {total_sents} sentences, {total_toks} tokens")
    write_csv(merged_raw, PROC_DIR / "merged.csv")
    print(f"    Saved → data/processed/merged.csv")

    # 4. BIO fix
    print("\n[4] Converting flat B-only → proper BIO (B-I)...")
    merged_bio = {sid: fix_bio(toks) for sid, toks in merged_raw.items()}
    fixes = sum(
        1
        for sid in merged_raw
        for (_, l1), (_, l2) in zip(merged_raw[sid], merged_bio[sid])
        if l1 != l2
    )
    print(f"    Labels changed B→I: {fixes}")
    write_csv(merged_bio, PROC_DIR / "merged_bio.csv")
    print(f"    Saved → data/processed/merged_bio.csv")

    # 5. Stats on merged BIO
    print("\n[5] Merged dataset statistics:")
    print_stats("MERGED", merged_bio)

    # 6. Split
    print("\n[6] Stratified 80/10/10 split...")
    train, val, test = stratified_split(merged_bio)
    write_csv(train, PROC_DIR / "train.csv")
    write_csv(val,   PROC_DIR / "val.csv")
    write_csv(test,  PROC_DIR / "test.csv")
    print_stats("TRAIN", train)
    print_stats("VAL",   val)
    print_stats("TEST",  test)

    print("\n" + "=" * 60)
    print("  Preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
