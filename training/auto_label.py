# -*- coding: utf-8 -*-
"""
Pelabel NER berbasis aturan + gazetteer untuk teks Banyumasan (bms.wikipedia).
Label: LOC (tempat), PER (orang), ORG (organisasi), TIME (waktu).
Skema: BIO (B-XXX / I-XXX / O).
"""
import csv, sys, re, json
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# 1. GAZETTEER & PENANDA (head nouns / pemicu)
# ---------------------------------------------------------------------------

# Kata kepala geografis -> jika diikuti kata berhuruf kapital, seluruh frasa = LOC
LOC_HEADS = {
    "kabupaten","kecamatan","kelurahan","kota","kotamadya","desa","dhusun","dusun",
    "provinsi","propinsi","kabupatén","negara","negeri","pulau","kapuloan","kepulauan",
    "gunung","pagunungan","pegunungan","kali","bengawan","tlaga","telaga","segara",
    "segoro","laut","tasik","samudra","samudera","tanjung","selat","teluk","ranu",
    "stasiun","setasiun","bandara","bandar","pelabuhan","palabuhan","pasar","alun-alun",
    "jalan","dalan","margi","kraton","keraton","candi","pura","masjid","mesjid","gereja",
    "wewengkon","wilayah","daérah","dataran","lembah","leng","benua",
}

# Entitas tempat berdiri sendiri (terkenal) -> LOC walau tanpa head noun
LOC_GAZ = {
    "indonesia","jawa","sumatera","sumatra","kalimantan","sulawesi","papua","bali",
    "madura","lombok","jakarta","yogyakarta","jogja","surabaya","semarang","bandung",
    "purwokerto","banyumas","cilacap","kebumen","purbalingga","banjarnegara","tegal",
    "brebes","pemalang","pekalongan","cirebon","wonosobo","magelang","klaten","solo",
    "surakarta","kudus","jepara","demak","rembang","blora","grobogan","sragen","boyolali",
    "sukoharjo","wonogiri","karanganyar","pati","batang","kendal","temanggung","salatiga",
    "malaysia","singapura","singapore","thailand","filipina","vietnam","kamboja","laos",
    "myanmar","brunei","timor","jepang","jepan","cina","china","tiongkok","korea","india",
    "arab","mesir","belanda","portugis","inggris","perancis","prancis","jerman","spanyol",
    "italia","rusia","amerika","afrika","eropa","eropah","asia","australia","antartika",
    "aceh","medan","palembang","lampung","makassar","manado","ambon","ternate","banten",
    "tahiti","mekah","mekkah","madinah","roma","london","paris","washington","beijing",
    "tokyo","seoul","bangkok","manila","kairo","baghdad","yerusalem","yerussalem",
    "nusantara","sunda","melayu","jawa tengah","jawa barat","jawa timur",
}

# Penanda organisasi
ORG_HEADS = {
    "universitas","institut","sekolah","sma","smp","smk","sd","madrasah","akademi",
    "perusahaan","perseroan","pt","cv","ud","koperasi","yayasan","lembaga","badan",
    "departemen","departemén","kementerian","kementrian","direktorat","dinas","kantor",
    "partai","organisasi","perserikatan","persatuan","ikatan","himpunan","komite",
    "klub","kesebelasan","tim","grup","band","majelis","majlis","kerajaan","karaton",
    "kesultanan","kasultanan","kasunanan","kadipaten","kabupatian","republik","komisi",
    "pemerintah","pamarentah","pamaréntah","gereja","keuskupan","sinode","ormas",
}
ORG_GAZ = {
    "pertamina","wikipedia","wikimedia","unesco","unicef","pbb","asean","fifa","uefa",
    "google","microsoft","facebook","twitter","nasa","who","gojek","tokopedia",
    "garuda","telkom","bni","bri","bca","mandiri","pln","kai","tvri","rri","tni","polri",
    "nu","muhammadiyah","golkar","pdip","gerindra","demokrat","pks","pkb","ppp","pan",
    "majapahit","mataram","singasari","singosari","sriwijaya","sriwijoyo","kediri",
    "pajang","kahuripan","tarumanagara","kutai","galuh","pajajaran",
}

# Penanda waktu
TIME_MONTHS = {
    "januari","februari","pebruari","maret","april","mei","juni","juli","agustus",
    "agustos","september","oktober","nopember","november","desember","desembér",
    # bulan jawa
    "sura","sapar","mulud","bakda","jumadilawal","jumadilakir","rejeb","ruwah","pasa",
    "sawal","sela","besar","saban","syawal","rajab","sya'ban","ramadan","ramadhan",
    "muharram","safar","dzulhijjah","dzulkaidah",
}
TIME_DAYS = {
    "senen","senin","selasa","slasa","rebo","rabu","kemis","kamis","jemuwah","jumat",
    "jumuah","setu","sabtu","minggu","ahad","akad","minggon",
    # pasaran jawa
    "legi","pahing","pon","wage","kliwon","kliwon",
}
TIME_TRIGGERS = {
    "taun","tahun","warsa","abad","kurun","windu","wulan","sasi","sasih","dina","dinten",
    "tanggal","tanggalan","tgl","jam","menit","detik","abat","mangsa","era","zaman","jaman",
}

# Gelar / pemicu nama orang -> kata kapital sesudahnya = PER
PER_TITLES = {
    "presiden","wakil","gubernur","bupati","walikota","wali","raja","ratu","prabu",
    "sultan","sunan","susuhunan","pangeran","pangéran","adipati","panembahan","ki","nyi",
    "kyai","kiai","kiyai","raden","mas","den","gusti","sri","sang","syekh","syeh","syaikh",
    "haji","hajjah","nabi","rasul","wali","santo","santa","sahabat","imam","ulama",
    "jenderal","jendral","letnan","kolonel","kapten","mayor","sersan","laksamana",
    "dokter","dr","prof","profesor","ir","drs","dra","st","ki","nyai","eyang","mbah",
    "pak","bapak","ibu","bu","tuan","nyonya","mister","mr","mrs","miss","sir","lord",
    "kaisar","kanjeng","tumenggung","tumenggong","demang","ngabehi","menteri","mentri",
    "patih","senopati","senapati","mahapatih",
}

# Stopword huruf-besar yang BUKAN entitas (awal kalimat / kata fungsi / arah)
NON_ENTITY_CAP = {
    "pada","nang","ning","ana","ono","iki","kiye","kuwe","kuwi","sing","kang","saka",
    "sekang","maring","karo","lan","utawa","banjur","terus","nuli","mula","dadi","amarga",
    "merga","ngandhut","wektu","wancine","yakuwe","yaiku","yaitu","contone","conto","tuladha",
    "deleng","delengen","pranala","referensi","prakata","pambuka","sejarah","geografi","geografis",
    "transportasi","ekonomi","budaya","pendidikan","pariwisata","wilayah","daftar","catetan",
    "catatan","sumber","pustaka","kapustakan","jaba","njaba","gambar","tabel","no","nomer",
    "utara","kidul","wetan","kulon","lor","selatan","timur","barat","tengah","tengen",
    "uga","kabeh","kabéh","akeh","akéh","ya","iya","ora","aja","mung","wae","baé","luwih",
    "paling","banget","temen","kuwé","kéné","kana","kono","ngisor","nduwur","dhuwur",
}

ABBR = {"st.","no.","dr.","ir.","drs.","prof.","kab.","kec.","gg.","jl.","jln.","tgl.","hal.","s.","a.","m."}

# ---------------------------------------------------------------------------
# 2. SENTENCE SPLIT & TOKENIZER
# ---------------------------------------------------------------------------
def split_sentences(text):
    text = text.replace("\r", "\n")
    # buang sisa tag html
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    parts = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # pecah pada . ! ? yang diikuti spasi+huruf besar atau akhir
        buf = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", line)
        for s in buf:
            s = s.strip()
            if len(s) >= 2:
                parts.append(s)
    return parts

TOKEN_RE = re.compile(r"[A-Za-z\u00C0-\u017F]+(?:[-'][A-Za-z\u00C0-\u017F]+)*|\d+(?:[.,]\d+)*|[^\sA-Za-z0-9]")
def tokenize(sent):
    return TOKEN_RE.findall(sent)

def is_cap(tok):
    return bool(tok) and tok[0].isupper() and any(c.isalpha() for c in tok)

# ---------------------------------------------------------------------------
# 3. TAGGER
# ---------------------------------------------------------------------------
def tag_sentence(tokens):
    n = len(tokens)
    labels = ["O"] * n
    low = [t.lower() for t in tokens]
    i = 0
    while i < n:
        tok, lt = tokens[i], low[i]

        # --- TIME: angka tahun 4 digit (1000-2099), bukan koordinat/ukuran ---
        COORD = {"'", "’", "ʼ", "\"", "”", "“", "°", "º", "%"}
        if re.fullmatch(r"\d{4}", tok) and 1000 <= int(tok) <= 2099:
            window = tokens[i+1:i+4]
            if not any(w in COORD for w in window):
                labels[i] = "B-TIME"; i += 1; continue
        # --- TIME: pemicu / bulan / hari ---
        if lt in TIME_TRIGGERS or lt in TIME_MONTHS or lt in TIME_DAYS:
            labels[i] = "B-TIME"; j = i + 1
            # rangkai elemen waktu berturut: angka, bulan, hari, pasaran, tahun
            while j < n and (re.fullmatch(r"\d{1,4}", tokens[j]) or low[j] in TIME_MONTHS
                             or low[j] in TIME_DAYS or low[j] in TIME_TRIGGERS):
                labels[j] = "I-TIME"; j += 1
            i = j; continue

        # --- PER: gelar diikuti nama kapital ---
        if lt in PER_TITLES and not (tok.isupper() and len(tok) > 1) \
           and i + 1 < n and is_cap(tokens[i+1]) and low[i+1] not in NON_ENTITY_CAP:
            labels[i] = "B-PER"; j = i + 1
            while j < n and is_cap(tokens[j]) and low[j] not in NON_ENTITY_CAP \
                  and low[j] not in LOC_HEADS and low[j] not in ORG_HEADS:
                labels[j] = "I-PER"; j += 1
            i = j; continue

        # --- ORG: kata kepala organisasi + nama kapital ---
        if lt in ORG_HEADS:
            if i + 1 < n and is_cap(tokens[i+1]) and low[i+1] not in NON_ENTITY_CAP:
                labels[i] = "B-ORG"; j = i + 1
                while j < n and (is_cap(tokens[j]) or low[j] in {"lan","dan","kang","sing"}) \
                      and low[j] not in NON_ENTITY_CAP:
                    labels[j] = "I-ORG"; j += 1
                i = j; continue
            elif is_cap(tok):  # PT, Universitas berdiri sendiri kapital
                labels[i] = "B-ORG"; i += 1; continue
        # --- ORG: gazetteer ---
        if lt in ORG_GAZ:
            labels[i] = "B-ORG"; i += 1; continue

        # --- LOC: kata kepala geografis + nama kapital ---
        if lt in LOC_HEADS:
            if i + 1 < n and is_cap(tokens[i+1]) and low[i+1] not in NON_ENTITY_CAP:
                labels[i] = "B-LOC"; j = i + 1
                while j < n and is_cap(tokens[j]) and low[j] not in NON_ENTITY_CAP \
                      and low[j] not in LOC_HEADS:
                    labels[j] = "I-LOC"; j += 1
                i = j; continue
            elif is_cap(tok):
                labels[i] = "B-LOC"; i += 1; continue
        # --- LOC: gazetteer (boleh frasa, mis. "Jawa Tengah") ---
        if lt in LOC_GAZ and is_cap(tok):
            labels[i] = "B-LOC"; j = i + 1
            # gabung arah/penjelas berikutnya bila kapital & gazetteer
            while j < n and is_cap(tokens[j]) and (low[j] in LOC_GAZ or low[j] in
                  {"tengah","barat","timur","kidul","lor","wetan","kulon","selatan",
                   "utara","tenggara","kulwetan","tlatah"}):
                labels[j] = "I-LOC"; j += 1
            i = j; continue

        i += 1
    return labels

# ---------------------------------------------------------------------------
# 4. PIPELINE ke seluruh file
# ---------------------------------------------------------------------------
def spans_from_bio(tokens, labels):
    ents, cur, ctype = [], [], None
    for t, l in zip(tokens, labels):
        if l.startswith("B-"):
            if cur: ents.append((" ".join(cur), ctype))
            cur, ctype = [t], l[2:]
        elif l.startswith("I-") and cur:
            cur.append(t)
        else:
            if cur: ents.append((" ".join(cur), ctype)); cur, ctype = [], None
    if cur: ents.append((" ".join(cur), ctype))
    return ents

def main():
    src = "/mnt/user-data/uploads/wiki_map_bms_text_only.csv"
    texts = []
    with open(src, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row: texts.append(row[0])

    conll_path = "/mnt/user-data/outputs/bms_ner_conll.txt"
    sent_path  = "/mnt/user-data/outputs/bms_ner_sentences.csv"
    tok_path   = "/mnt/user-data/outputs/bms_ner_tokens.csv"

    from collections import Counter
    counts = Counter()
    n_sent = 0; n_tok = 0

    fc = open(conll_path, "w", encoding="utf-8")
    fs = open(sent_path, "w", newline="", encoding="utf-8")
    ft = open(tok_path, "w", newline="", encoding="utf-8")
    ws = csv.writer(fs); wt = csv.writer(ft)
    ws.writerow(["doc_id","sent_id","sentence","entities"])
    wt.writerow(["doc_id","sent_id","token_id","token","label"])

    for doc_id, text in enumerate(texts):
        for sent_id, sent in enumerate(split_sentences(text)):
            toks = tokenize(sent)
            if not toks:
                continue
            labels = tag_sentence(toks)
            n_sent += 1; n_tok += len(toks)
            for k, (tk, lb) in enumerate(zip(toks, labels)):
                fc.write(f"{tk}\t{lb}\n")
                wt.writerow([doc_id, sent_id, k, tk, lb])
                if lb != "O" and lb.startswith("B-"):
                    counts[lb[2:]] += 1
            fc.write("\n")
            ents = spans_from_bio(toks, labels)
            ws.writerow([doc_id, sent_id, sent,
                         json.dumps(ents, ensure_ascii=False)])

    fc.close(); fs.close(); ft.close()

    print("SELESAI")
    print("Dokumen :", len(texts))
    print("Kalimat :", n_sent)
    print("Token   :", n_tok)
    print("Entitas ditemukan:")
    for k in ["LOC","PER","ORG","TIME"]:
        print(f"  {k}: {counts[k]}")

if __name__ == "__main__":
    main()