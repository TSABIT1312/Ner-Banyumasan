"""
Inference wrapper for the trained Banyumasan NER CRF model.

Exposes predict_ner(text) — same output contract as the rule-based
function it is designed to replace in app.py.

Output schema (list of dicts):
    {
        "token":      str,   # original surface form
        "label":      str,   # PER | LOC | ORG | MISC | O
        "confidence": float  # marginal probability of the predicted label
    }

Strategy: hybrid inference.
  1. CRF predicts all tokens — trusted for PER, ORG, MISC.
  2. auto_label.tag_sentence (rule-based gazetteers) runs in parallel.
  3. Where CRF says O but auto_label says TIME or LOC, auto_label wins
     (CRF has too few TIME/LOC training examples to be reliable).

Usage (standalone smoke test):
    python app/predict.py
"""

import re
import json
import sys
import joblib
from pathlib import Path

# _PROJECT is the repository root — one level above this file's directory (app/)
_PROJECT = Path(__file__).parent.parent
_TRAIN   = _PROJECT / "training"
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from features   import sent_to_features  # noqa: E402
from auto_label import tag_sentence      # noqa: E402

_MODEL_DIR = _PROJECT / "models"

# ── Load model once at import time ────────────────────────────────────────────
_crf        = joblib.load(_MODEL_DIR / "crf_ner.joblib")
_label_info = json.loads((_MODEL_DIR / "label_encoder.json").read_text(encoding="utf-8"))
_all_labels = _label_info["labels"]

# Same tokenizer as auto_label.py — splits punctuation off words so gazetteer
# features match correctly (e.g. "Senen," → ["Senen", ","])
_TOKEN_RE = re.compile(
    r"[A-Za-zÀ-ſ]+(?:[-'][A-Za-zÀ-ſ]+)*"
    r"|\d+(?:[.,]\d+)*"
    r"|[^\sA-Za-z0-9]"
)

# Confidence assigned to auto_label fallback results (rule-based, no probability)
_RULE_CONF = 0.85


def predict_ner(text: str) -> list[dict]:
    """
    Hybrid CRF + rule-based NER inference.

    CRF is used as primary predictor. For TIME and LOC where the CRF
    predicts O, the rule-based auto_label output is used as fallback.
    """
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return []

    # ── CRF pass ──────────────────────────────────────────────────────────────
    sent      = [(t, "O") for t in tokens]
    feats     = sent_to_features(sent)
    crf_labels    = _crf.predict([feats])[0]
    crf_marginals = _crf.predict_marginals([feats])[0]

    # ── Rule-based pass (auto_label) ──────────────────────────────────────────
    rule_labels = tag_sentence(tokens)   # returns list of BIO strings

    # ── Merge (mirrors merge_labels.py training strategy exactly) ─────────────
    # auto_label non-O always wins; CRF fills in where auto_label says O.
    # This keeps inference consistent with how the model was trained.
    results = []
    for tok, crf_lbl, marg, rule_lbl in zip(tokens, crf_labels, crf_marginals, rule_labels):
        if rule_lbl != "O":
            clean = rule_lbl.split("-", 1)[1]
            conf  = _RULE_CONF
        elif crf_lbl != "O":
            clean = crf_lbl.split("-", 1)[1]
            conf  = round(float(marg.get(crf_lbl, 0.0)), 4)
        else:
            clean = "O"
            conf  = round(float(marg.get("O", 1.0)), 4)

        results.append({"token": tok, "label": clean, "confidence": conf})

    return results


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TESTS = [
        "Dina Senen, tanggal 17 Agustus 1945, Indonesia merdeka saka penjajah.",
        "Wiwit sasi Januari nganti Maret, udan teka saben dina tanpa lèrèn.",
        "Taun 2023 akeh wong Banyumas sing mulai nulis neng platform digital.",
        "Wingi esuk, sekitar jam pitu, Sépul mangkat sekolah karo batire.",
        "Agus karo Dewi nonton wayang ning Sokaraja bengi.",
        "Dalan menyang Purwokerto ditutup amarga ana longsor gedhe.",
        "Kali Serayu mili saka Gunung Slamet nganti tekan segara kidul.",
        "Pak Joko minangka ketua DPRD Cilacap wiwit taun wingi.",
        "Slamet lunga menyang Banyumas wingi karo Budi.",
        "Gurusiana dadi papan kanggo ngudhalaken rasa lan blajar nulis.",
    ]

    print("=" * 60)
    print("  predict.py — hybrid CRF + rule-based smoke test")
    print("=" * 60)
    for text in TESTS:
        results = predict_ner(text)
        print(f"\nINPUT : {text}")
        entities = [r for r in results if r["label"] != "O"]
        if entities:
            for r in entities:
                print(f"  [{r['label']:5s}] {r['token']:22s}  conf={r['confidence']:.2%}")
        else:
            print("  (no entities detected)")
