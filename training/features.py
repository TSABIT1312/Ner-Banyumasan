"""
CRF feature extraction for Banyumasan NER — v3.

Changes from v2:
  - TIME_MONTHS   : Indonesian + Javanese Islamic calendar month names
  - TIME_DAYS     : Indonesian + Banyumasan day names + Javanese pasaran
  - TIME_TRIGGERS : temporal head words (taun, wulan, dina, tanggal, abad …)
  - ORG_GAZ       : known organisation proper names from auto_label.py
  - word.is_year  : boolean True when token matches 4-digit year 1000-2099
  - lex.time_*    : month / day / trigger membership for current token
  - lex.org_gaz   : org gazetteer membership for current token
  - context:      : -1/+1 time_trigger and org_gaz signals added to windows

LOC_GAZETTEER and TOKEN_FREQ remain module-level globals written by
train_crf.py before feature extraction and auto-loaded on import.
"""

import re
import json
from pathlib import Path

# ── Auto-load runtime data from models/ ───────────────────────────────────────
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


LOC_GAZETTEER: set = set(_load_json(_MODELS_DIR / "loc_gazetteer.json", []))
TOKEN_FREQ: dict   = _load_json(_MODELS_DIR / "token_freq.json", {})

# ── TIME gazetteers (from auto_label.py) ─────────────────────────────────────

TIME_MONTHS = {
    "januari","februari","pebruari","maret","april","mei","juni","juli",
    "agustus","agustos","september","oktober","nopember","november",
    "desember","desembér",
    # Javanese Islamic calendar months
    "sura","sapar","mulud","bakda","jumadilawal","jumadilakir","rejeb",
    "ruwah","pasa","sawal","sela","besar","saban","syawal","rajab",
    "sya'ban","ramadan","ramadhan","muharram","safar","dzulhijjah","dzulkaidah",
}

TIME_DAYS = {
    # Indonesian days
    "senen","senin","selasa","slasa","rebo","rabu","kemis","kamis",
    "jemuwah","jumat","jumuah","setu","sabtu","minggu","ahad","akad","minggon",
    # Javanese pasaran cycle
    "legi","pahing","pon","wage","kliwon",
}

TIME_TRIGGERS = {
    "taun","tahun","warsa","abad","kurun","windu","wulan","sasi","sasih",
    "dina","dinten","tanggal","tanggalan","tgl","jam","menit","detik",
    "abat","mangsa","era","zaman","jaman",
}

# ── ORG gazetteer (from auto_label.py) ───────────────────────────────────────

ORG_GAZ = {
    "pertamina","wikipedia","wikimedia","unesco","unicef","pbb","asean",
    "fifa","uefa","google","microsoft","facebook","twitter","nasa","who",
    "gojek","tokopedia","garuda","telkom","bni","bri","bca","mandiri",
    "pln","kai","tvri","rri","tni","polri","nu","muhammadiyah","golkar",
    "pdip","gerindra","demokrat","pks","pkb","ppp","pan",
    "majapahit","mataram","singasari","sriwijaya","sriwijoyo","kediri",
    "pajang","kahuripan","tarumanagara","kutai","galuh","pajajaran",
}

# ── Domain lexicons ───────────────────────────────────────────────────────────

PERSON_DICT = {
    "slamet","budi","siti","bambang","rini","agus","dewi","joko","ani",
    "hendra","wahyu","rina","tono","wati","arif","dian","yanto","sri",
    "kariadi","sudirman","sarinem","parman","kasinem","paimin","sutrisno",
    "eko","heri","yuli","wawan","tanto","ratna","endah","teguh","mulyono",
    "suparno","darmo","kasino","tukiman","supomo","harjo","wiryo","suwanto",
    "gunawan","hartono","prasetyo","marno","tarno","kirno","sarno","parno",
    "poniman","suminah","ngadinem","paijo","waginem","waginah","ningsih",
    "supri","suprianto","supriyadi","subekti","sumarno","sumarni","sumini",
    "santoso","haryono","purwanto","purwati","wardoyo","wardani","lestari",
    "indah","indriani","setiawan","setiawati","nugroho","wibowo","wibawati",
    "hamidin","hamizan","sari","fitri","nuri","yusuf","ahmad","ali","hasan",
    "aminah","fatimah","khasanah","rohmah","mulyani","setiadi","prayitno",
    # Characters specific to training stories
    "sépul","nasir","brondhol","yanto","rida",
}

LOCATION_DICT = {
    "banyumas","purwokerto","cilacap","kebumen","purbalingga","banjarnegara",
    "sokaraja","ajibarang","wangon","sumpiuh","rawalo","lumbir","jatilawang",
    "kalibagor","tambak","kemranjen","srandakan","kutasari",
    "baturraden","kaliori","pekuncen","patikraja","karanglewas","sumbang",
    "gumelar","leksono","kedungbanteng","karanganyar","majenang",
    "gandrungmangu","sampang","adipala","binangun","nusawungu","kesugihan",
    "sidareja","jeruklegi","kawunganten","bantarsari","karangpucung",
    "cimanggu","dayeuhluhur","purwojati","kalimanah","bobotsari",
    "jawa","tengah","indonesia","jateng","solo","semarang","yogyakarta",
    "jogja","magelang","temanggung","wonosobo","brebes","tegal",
    "pemalang","pekalongan","batang","kendal","demak","kudus","jepara",
    "purwakerta","penginyongan","karesidenan","banyumasan","malioboro",
}

ORG_DICT = {
    "pemda","pemkab","dprd","puskesmas","rsud","bpjs","kpu","polsek",
    "koramil","dinas","bumn","koperasi","bumdes","polres","polda","polri",
    "bupati","camat","pemerintah","pemkot","dpd","dpr","mpr",
    "kemenkes","kemendikbud","kemenag","kemenkeu","kemendagri",
    "bank","bri","bni","bca","mandiri","btn","pos","telkom","pln",
    "pertamina","pgn","pdam","damkar","satpol","dishub","dinkes",
    "bappeda","bawaslu","kpk","kejaksaan","pengadilan","mahkamah",
    "masjid","gereja","pesantren","ponpes","sma","smk","smp","sd","tk",
    "universitas","akademi","sekolah","lembaga","yayasan","partai",
    "posyandu","polindes","poskamling","kua","kantor","gurusiana",
    "mediaguru","madrasah",
}

PERSON_TITLES = {
    "pak","bu","mas","mbak","kang","yu","bapak","ibu","dr","prof",
    "haji","hj","drs","ir","ustadz","kyai","ust","kh","rm","rr","raden",
    "den","tuan","nona","nyonya","gus","ning","ndoro","dalem",
}

# Extended: added Banyumasan spatial verbs and position words
LOC_PREPOSITIONS = {
    "ning","nang","menyang","saka","nyang","mring","tekan","neng",
    "dari","ke","di","wilayah","daerah","desa","kecamatan","kabupaten",
    "kota","provinsi","kelurahan","kampung","kawasan","sekitar",
    "ngarep","njero","ndhuwur","ngisor","mburi","pinggir","tengah",
    "wetan","kulon","lor","kidul","ndhep","ngidul","ngulon",
    "mlebet","medal","munggah","mudun","mlaku","lunga","teka","mulih",
    "ngepel","ngepél","resik","nyapu","resiki",
}

ORG_CONTEXT_WORDS = {
    "ketua","anggota","kepala","direktur","sekretaris","pegawe","pegawai",
    "staf","bendahara","wakil","pengurus","pimpinan","kabid","kasubid",
    "kasi","koordinator","instansi","lembaga","organisasi","kantor",
    "pemimpin","karyawan","petugas","aparatur","kadher","kader",
}

# Common Banyumasan inflectional suffixes for stem matching
_SUFFIXES = ("ne", "e", "ke", "mu", "an", "ku", "nipun", "é")

# Pre-compiled year pattern
_YEAR_RE = re.compile(r"^\d{4}$")


# ── Helper functions ──────────────────────────────────────────────────────────

def word_shape(word: str) -> str:
    """
    Map each char to X (upper), x (lower), d (digit), or itself,
    then collapse consecutive identical chars.

    'Banyumas' → 'Xx', 'DPRD' → 'X', '2023' → 'd', 'alun-alun' → 'x-x'
    """
    if not word:
        return ""
    chars = []
    for c in word:
        if c.isupper():   chars.append("X")
        elif c.islower(): chars.append("x")
        elif c.isdigit(): chars.append("d")
        else:             chars.append(c)
    collapsed = chars[0]
    for c in chars[1:]:
        if c != collapsed[-1]:
            collapsed += c
    return collapsed


def get_stem(word: str) -> str:
    """Strip the longest matching Banyumasan inflectional suffix (min stem=3)."""
    wl = word.lower()
    for suf in sorted(_SUFFIXES, key=len, reverse=True):
        if wl.endswith(suf) and len(wl) - len(suf) >= 3:
            return wl[: -len(suf)]
    return wl


def get_freq_bucket(wl: str) -> str:
    n = TOKEN_FREQ.get(wl, 0)
    if n == 0:  return "unseen"
    if n == 1:  return "hapax"
    if n <= 5:  return "rare"
    if n <= 20: return "medium"
    return "frequent"


def is_year(word: str) -> bool:
    """True if token looks like a calendar year (1000–2099)."""
    return bool(_YEAR_RE.match(word)) and 1000 <= int(word) <= 2099


# ── Feature extractor ─────────────────────────────────────────────────────────

def word_features(sent: list[tuple], i: int) -> dict:
    word = sent[i][0]
    wl   = word.lower()
    stem = get_stem(word)

    feats: dict = {
        "bias": 1.0,

        # ── Exact form ────────────────────────────────────────
        "word.lower":  wl,

        # ── Prefix (2-5) ──────────────────────────────────────
        "word[:2]":    word[:2],
        "word[:3]":    word[:3],
        "word[:4]":    word[:4],
        "word[:5]":    word[:5],

        # ── Suffix (2-5) ──────────────────────────────────────
        "word[-2:]":   word[-2:],
        "word[-3:]":   word[-3:],
        "word[-4:]":   word[-4:],
        "word[-5:]":   word[-5:],

        # ── Word shape ────────────────────────────────────────
        "word.shape":              word_shape(word),
        "word.isupper":            word.isupper(),
        "word.istitle":            word.istitle(),
        "word.islower":            word.islower(),
        "word.init_cap":           word[0].isupper() if word else False,
        "word.all_caps":           word.isupper() and len(word) > 1,
        "word.mixed_case":         (any(c.isupper() for c in word)
                                    and any(c.islower() for c in word)),
        "word.has_internal_upper": any(c.isupper() for c in word[1:]),

        # ── Other shape / type flags ──────────────────────────
        "word.isdigit":    word.isdigit(),
        "word.is_year":    is_year(word),
        "word.has_hyphen": "-" in word,
        "word.has_digit":  any(c.isdigit() for c in word),
        "word.has_dot":    "." in word,
        "word.len_le2":    len(word) <= 2,
        "word.len_gt8":    len(word) > 8,

        # ── Token frequency ───────────────────────────────────
        "word.freq_bucket": get_freq_bucket(wl),

        # ── Lexicon membership ────────────────────────────────
        "lex.person":         wl   in PERSON_DICT,
        "lex.location":       wl   in LOCATION_DICT,
        "lex.org":            wl   in ORG_DICT,
        "lex.org_gaz":        wl   in ORG_GAZ,
        "lex.title":          wl   in PERSON_TITLES,
        "lex.loc_prep":       wl   in LOC_PREPOSITIONS,
        "lex.org_ctx":        wl   in ORG_CONTEXT_WORDS,
        "lex.loc_gazetteer":  wl   in LOC_GAZETTEER,
        "lex.stem_loc_gaz":   stem in LOC_GAZETTEER,
        # TIME signals
        "lex.time_month":     wl   in TIME_MONTHS,
        "lex.time_day":       wl   in TIME_DAYS,
        "lex.time_trigger":   wl   in TIME_TRIGGERS,
    }

    # ── Character bigrams ─────────────────────────────────────
    if len(wl) >= 2:
        for j in range(len(wl) - 1):
            feats[f"cbg:{wl[j:j+2]}"] = True

    # ── Character trigrams ────────────────────────────────────
    if len(wl) >= 3:
        for j in range(len(wl) - 2):
            feats[f"ctg:{wl[j:j+3]}"] = True

    # ── Context window -2 ─────────────────────────────────────
    if i > 1:
        pp  = sent[i - 2][0]
        ppl = pp.lower()
        feats.update({
            "-2:word.lower":        ppl,
            "-2:word.istitle":      pp.istitle(),
            "-2:lex.title":         ppl in PERSON_TITLES,
            "-2:lex.loc_prep":      ppl in LOC_PREPOSITIONS,
            "-2:lex.loc_gazetteer": ppl in LOC_GAZETTEER,
            "-2:lex.time_trigger":  ppl in TIME_TRIGGERS,
            "-2:word.is_year":      is_year(pp),
            "-2:word[-3:]":         pp[-3:],
        })

    # ── Context window -1 ─────────────────────────────────────
    if i > 0:
        prev  = sent[i - 1][0]
        prevl = prev.lower()
        feats.update({
            "-1:word.lower":        prevl,
            "-1:word.istitle":      prev.istitle(),
            "-1:word.isupper":      prev.isupper(),
            "-1:word.init_cap":     prev[0].isupper() if prev else False,
            "-1:word.is_year":      is_year(prev),
            "-1:lex.person":        prevl in PERSON_DICT,
            "-1:lex.location":      prevl in LOCATION_DICT,
            "-1:lex.org":           prevl in ORG_DICT,
            "-1:lex.org_gaz":       prevl in ORG_GAZ,
            "-1:lex.title":         prevl in PERSON_TITLES,
            "-1:lex.loc_prep":      prevl in LOC_PREPOSITIONS,
            "-1:lex.org_ctx":       prevl in ORG_CONTEXT_WORDS,
            "-1:lex.loc_gazetteer": prevl in LOC_GAZETTEER,
            "-1:lex.time_trigger":  prevl in TIME_TRIGGERS,
            "-1:lex.time_month":    prevl in TIME_MONTHS,
            "-1:lex.time_day":      prevl in TIME_DAYS,
            "-1:word[-3:]":         prev[-3:],
            "-1:word[:3]":          prev[:3],
            "-1:word.freq_bucket":  get_freq_bucket(prevl),
        })
    else:
        feats["BOS"] = True

    # ── Context window +1 ─────────────────────────────────────
    if i < len(sent) - 1:
        nxt  = sent[i + 1][0]
        nxtl = nxt.lower()
        feats.update({
            "+1:word.lower":        nxtl,
            "+1:word.istitle":      nxt.istitle(),
            "+1:word.isupper":      nxt.isupper(),
            "+1:word.is_year":      is_year(nxt),
            "+1:lex.location":      nxtl in LOCATION_DICT,
            "+1:lex.org":           nxtl in ORG_DICT,
            "+1:lex.org_gaz":       nxtl in ORG_GAZ,
            "+1:lex.loc_prep":      nxtl in LOC_PREPOSITIONS,
            "+1:lex.org_ctx":       nxtl in ORG_CONTEXT_WORDS,
            "+1:lex.loc_gazetteer": nxtl in LOC_GAZETTEER,
            "+1:lex.time_trigger":  nxtl in TIME_TRIGGERS,
            "+1:lex.time_month":    nxtl in TIME_MONTHS,
            "+1:lex.time_day":      nxtl in TIME_DAYS,
            "+1:word[-3:]":         nxt[-3:],
            "+1:word[:3]":          nxt[:3],
            "+1:word.freq_bucket":  get_freq_bucket(nxtl),
        })
    else:
        feats["EOS"] = True

    # ── Context window +2 ─────────────────────────────────────
    if i < len(sent) - 2:
        nn  = sent[i + 2][0]
        nnl = nn.lower()
        feats.update({
            "+2:word.lower":        nnl,
            "+2:word.istitle":      nn.istitle(),
            "+2:word.is_year":      is_year(nn),
            "+2:lex.loc_gazetteer": nnl in LOC_GAZETTEER,
            "+2:lex.time_trigger":  nnl in TIME_TRIGGERS,
        })

    return feats


def sent_to_features(sent: list[tuple]) -> list[dict]:
    return [word_features(sent, i) for i in range(len(sent))]


def sent_to_labels(sent: list[tuple]) -> list[str]:
    return [label for _, label in sent]
