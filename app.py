import streamlit as st
import pandas as pd
import re
import time
import json
from collections import Counter
from predict import predict_ner

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NER Banyumasan · Kelompok Ardhika",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Batik-inspired warm dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Code+Pro:wght@400;600&display=swap');

:root {
    --cream:   #F5EDD8;
    --brown:   #3B1F0E;
    --dark:    #1A0C04;
    --gold:    #C8922A;
    --rust:    #8B3A1A;
    --sage:    #4A6741;
    --teal:    #2D6B6B;
    --warm-mid:#7A4A2A;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'Libre Baskerville', Georgia, serif;
    background-color: var(--dark) !important;
    color: var(--cream) !important;
}

.stApp { background-color: var(--dark) !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px; }

/* ── HERO HEADER ── */
.hero {
    background: linear-gradient(135deg, #2A1205 0%, #3B1F0E 40%, #5C2E10 70%, #2A1205 100%);
    border: 2px solid var(--gold);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background-image: repeating-linear-gradient(
        45deg,
        transparent,
        transparent 18px,
        rgba(200,146,42,0.06) 18px,
        rgba(200,146,42,0.06) 19px
    );
}
.hero-badge {
    display: inline-block;
    background: var(--gold);
    color: var(--dark);
    font-family: 'Source Code Pro', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 900;
    color: var(--cream);
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
    letter-spacing: -1px;
}
.hero h1 span { color: var(--gold); }
.hero p {
    color: #C9B99A;
    font-size: 1.05rem;
    margin: 0.8rem 0 0 0;
    font-style: italic;
    max-width: 640px;
    margin-left: auto;
    margin-right: auto;
}
.hero-divider {
    width: 80px; height: 3px;
    background: var(--gold);
    margin: 1.2rem auto 0;
    border-radius: 2px;
}

/* ── SECTION HEADERS ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: var(--gold);
    border-left: 4px solid var(--gold);
    padding-left: 12px;
    margin-bottom: 1rem;
    letter-spacing: 0.5px;
}

/* ── CARD ── */
.card {
    background: rgba(59,31,14,0.45);
    border: 1px solid rgba(200,146,42,0.3);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}

/* ── ENTITY TAGS ── */
.entity-highlight {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 5px;
    font-weight: 700;
    font-size: 0.95rem;
    margin: 2px;
}
.tag-PER  { background: rgba(139,58,26,0.35); border: 1.5px solid #C0552A; color: #F5A07A; }
.tag-LOC  { background: rgba(45,107,107,0.35); border: 1.5px solid #3A9090; color: #7ADADA; }
.tag-ORG  { background: rgba(122,74,42,0.35); border: 1.5px solid #B07040; color: #E0B888; }
.tag-MISC { background: rgba(90,60,130,0.35); border: 1.5px solid #9060C0; color: #C8A0F0; }
.tag-TIME { background: rgba(74,103,65,0.35);  border: 1.5px solid #5E9A50; color: #A3D98E; }
.tag-O    { background: transparent; color: var(--cream); }

.entity-label {
    font-family: 'Source Code Pro', monospace;
    font-size: 0.62rem;
    vertical-align: super;
    margin-left: 2px;
    opacity: 0.85;
}

/* ── TABLE ── */
.result-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
    margin-top: 0.8rem;
}
.result-table th {
    background: rgba(200,146,42,0.15);
    color: var(--gold);
    font-family: 'Source Code Pro', monospace;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 8px 14px;
    border-bottom: 2px solid rgba(200,146,42,0.4);
    text-align: left;
}
.result-table td {
    padding: 8px 14px;
    border-bottom: 1px solid rgba(200,146,42,0.1);
    color: var(--cream);
}
.result-table tr:hover td { background: rgba(200,146,42,0.05); }

/* ── CONFIDENCE BAR ── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    height: 10px;
    width: 100%;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #8B3A1A, #C8922A);
    transition: width 0.6s ease;
}

/* ── LEGEND ── */
.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-right: 16px;
    font-size: 0.85rem;
    color: #C9B99A;
}
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
}

/* ── STREAMLIT OVERRIDES ── */
.stTextArea textarea {
    background: rgba(26,12,4,0.7) !important;
    border: 1.5px solid rgba(200,146,42,0.4) !important;
    border-radius: 10px !important;
    color: var(--cream) !important;
    font-family: 'Libre Baskerville', serif !important;
    font-size: 0.95rem !important;
    caret-color: var(--gold);
}
.stTextArea textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(200,146,42,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #8B3A1A, #C8922A) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Source Code Pro', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stSelectbox select, div[data-baseweb="select"] {
    background: rgba(26,12,4,0.7) !important;
    border-color: rgba(200,146,42,0.4) !important;
    color: var(--cream) !important;
    border-radius: 8px !important;
}
.stFileUploader {
    border: 1.5px dashed rgba(200,146,42,0.4) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
.stMetric { background: rgba(59,31,14,0.4); border-radius: 10px; padding: 0.8rem 1rem; }
.stMetric label { color: var(--gold) !important; font-family: 'Source Code Pro', monospace !important; font-size: 0.75rem !important; }
.stMetric [data-testid="stMetricValue"] { color: var(--cream) !important; font-family: 'Playfair Display', serif !important; }
div[data-testid="stTabs"] button {
    color: #C9B99A !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom-color: var(--gold) !important;
}
.stAlert { border-radius: 10px !important; }
hr { border-color: rgba(200,146,42,0.2) !important; }
.stSpinner > div { border-top-color: var(--gold) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KAMUS BANYUMASAN → INDONESIA
# ─────────────────────────────────────────────
BANYUMASAN_DICT = {
    # Kata ganti & sapaan
    "aku": "saya", "kowe": "kamu", "dheweke": "dia", "deweke": "dia",
    "awakedhewe": "kita", "awake": "dirinya", "pak": "bapak", "bu": "ibu",
    "mas": "mas", "mbak": "mbak", "kang": "kakak", "yu": "kakak",
    # Kata kerja umum
    "lunga": "pergi", "teka": "datang", "mangan": "makan", "ngombe": "minum",
    "turu": "tidur", "tangi": "bangun", "mlaku": "berjalan", "mlayu": "berlari",
    "ngomong": "berbicara", "krungu": "mendengar", "weruh": "melihat",
    "ngerti": "mengerti", "nonton": "menonton", "mbangun": "membangun",
    "nggawa": "membawa", "nukokke": "membelikan", "nukoki": "membelikan",
    "nukoku": "membeli", "tuku": "beli", "adol": "jual", "golek": "mencari",
    "nemu": "menemukan", "nggawe": "membuat", "ndeleng": "melihat",
    "nyambut": "bekerja", "kerja": "bekerja", "mulih": "pulang",
    "mangkat": "berangkat", "sinau": "belajar", "ngajar": "mengajar",
    "maca": "membaca", "nulis": "menulis", "takon": "bertanya",
    "njawab": "menjawab", "nyekel": "memegang", "nggowo": "membawa",
    "ngrungokke": "mendengarkan", "ngrungu": "mendengar",
    # Kata keterangan waktu
    "wingi": "kemarin", "sesuk": "besok", "saiki": "sekarang",
    "mau": "tadi", "bengi": "tadi malam", "esuk": "pagi",
    "awan": "siang", "sore": "sore", "wengi": "malam",
    "biyen": "dulu", "mengko": "nanti", "isuk": "pagi",
    "sedurunge": "sebelumnya", "sakwise": "setelah", "wis": "sudah",
    "durung": "belum", "isih": "masih", "lagek": "sedang/baru",
    "entar": "sebentar lagi",
    # Hari & waktu
    "dina": "hari", "senen": "Senin", "selasa": "Selasa", "rebo": "Rabu",
    "kemis": "Kamis", "jemuwah": "Jumat", "sabtu": "Sabtu", "ahad": "Minggu",
    "minggu": "minggu", "taun": "tahun", "wulan": "bulan",
    # Preposisi & konjungsi
    "menyang": "ke", "saka": "dari", "ning": "di", "karo": "dengan/bersama",
    "ambi": "dengan", "lan": "dan", "utawa": "atau", "nanging": "tetapi",
    "yen": "kalau/jika", "merga": "karena", "supaya": "supaya",
    "senajan": "meskipun", "nganti": "sampai", "kanggo": "untuk",
    "minangka": "sebagai", "wiwit": "sejak", "tekan": "sampai",
    "nang": "di", "nyang": "ke",
    # Kata sifat
    "apik": "bagus/baik", "elek": "jelek", "gedhe": "besar", "cilik": "kecil",
    "akeh": "banyak", "sithik": "sedikit", "adoh": "jauh", "cedhak": "dekat",
    "duwur": "tinggi", "endhek": "rendah", "anyar": "baru", "lawas": "lama",
    "panas": "panas", "adhem": "dingin", "abot": "berat", "entheng": "ringan",
    "cepet": "cepat", "alon": "pelan", "sugih": "kaya", "mlarat": "miskin",
    "seneng": "senang/suka", "susah": "susah/sedih", "sehat": "sehat",
    "lara": "sakit", "kesel": "lelah", "ngantuk": "mengantuk",
    "kuat": "kuat", "lemah": "lemah", "pinter": "pintar", "bodho": "bodoh",
    "becik": "baik", "ala": "buruk", "bener": "benar", "salah": "salah",
    "amba": "lebar/luas", "jero": "dalam", "cetek": "dangkal",
    # Kata benda umum
    "omah": "rumah", "uwong": "orang", "wong": "orang", "bocah": "anak",
    "wedok": "perempuan", "lanang": "laki-laki", "tuwa": "orang tua/tua",
    "enom": "muda", "dalan": "jalan", "banyu": "air", "geni": "api",
    "wit": "pohon", "lemah": "tanah", "langit": "langit", "srengenge": "matahari",
    "wulan": "bulan", "lintang": "bintang", "udan": "hujan", "angin": "angin",
    "segara": "laut", "kali": "sungai", "gunung": "gunung", "sawah": "sawah",
    "tegalan": "ladang", "alas": "hutan", "pasar": "pasar", "toko": "toko",
    "sekolah": "sekolah", "kantor": "kantor", "desa": "desa", "kutha": "kota",
    "jajan": "jajan/makanan kecil", "pangan": "makanan", "panganan": "makanan",
    "sandhangan": "pakaian", "buku": "buku", "motor": "motor", "mobil": "mobil",
    "wayang": "wayang", "gamelan": "gamelan",
    # Kata modal & partikel
    "arep": "akan/mau", "gelem": "mau/bersedia", "bisa": "bisa",
    "kudu": "harus", "perlu": "perlu", "wajib": "wajib", "ora": "tidak",
    "dudu": "bukan", "aja": "jangan", "ya": "ya", "iya": "iya",
    "ta": "kan/ya", "lho": "lho", "kok": "kok", "wae": "saja",
    "bakal": "akan", "mesti": "pasti", "kadhang": "kadang",
    "asring": "sering", "tansah": "selalu", "tau": "pernah",
    # Kata tanya
    "sapa": "siapa", "apa": "apa", "endi": "mana", "kapan": "kapan",
    "piye": "bagaimana", "pira": "berapa", "kenapa": "kenapa",
    "ngapa": "kenapa/ada apa",
    # Lain-lain
    "karo": "dengan/bersama", "nang": "di", "mring": "ke/kepada",
    "marang": "kepada", "dening": "oleh", "kabeh": "semua",
    "liyane": "lainnya", "liya": "lain", "padha": "sama/semua",
    "dhewe": "sendiri", "bareng": "bersama", "bebarengan": "bersama-sama",
}


def translate_banyumasan(text: str) -> tuple[str, int]:
    """
    Terjemahkan teks Banyumasan ke Bahasa Indonesia secara kata per kata.
    Mengembalikan (teks_terjemahan, jumlah_kata_diterjemahkan).
    """
    tokens = re.findall(r"[\w']+|[^\w\s]|\s+", text)
    translated = []
    count = 0
    for tok in tokens:
        if tok.strip() == "" or not tok.strip().isalpha():
            translated.append(tok)
            continue
        lookup = tok.lower().rstrip(".,!?;:")
        if lookup in BANYUMASAN_DICT:
            indo = BANYUMASAN_DICT[lookup]
            # Pertahankan huruf kapital awal
            if tok[0].isupper():
                indo = indo[0].upper() + indo[1:]
            translated.append(indo)
            count += 1
        else:
            translated.append(tok)
    return "".join(translated), count




def build_highlighted_html(results):
    COLORS = {
        "PER":  ("tag-PER",  "PER"),
        "LOC":  ("tag-LOC",  "LOC"),
        "ORG":  ("tag-ORG",  "ORG"),
        "MISC": ("tag-MISC", "MSC"),
        "TIME": ("tag-TIME", "TME"),
    }
    parts = []
    for r in results:
        tok, lbl = r["token"], r["label"]
        if lbl == "O":
            parts.append(f'<span class="entity-highlight tag-O">{tok}</span>')
        else:
            css, short = COLORS.get(lbl, ("tag-O", lbl))
            parts.append(
                f'<span class="entity-highlight {css}">'
                f'{tok}<sup class="entity-label">{short}</sup>'
                f'</span>'
            )
    return " ".join(parts)


# ─────────────────────────────────────────────
# SAMPLE SENTENCES
# ─────────────────────────────────────────────
SAMPLES = [
    # ── PER ──────────────────────────────────
    "Pak Joko minangka ketua DPRD Cilacap wiwit taun wingi.",
    "Slamet lunga menyang Banyumas wingi karo Budi.",
    "Bu Siti dadi guru kelas loro neng SD Negeri Sokaraja.",
    "Pak Hamidin dilantik dadi kepala desa neng Karangjana wiwit taun 2019.",
    "Dr. Bambang lan Prof. Rini sarujuk kanggo ngembangnaken riset basa Banyumasan.",
    "Kang Yanto nyambut gawe neng kantor kecamatan Ajibarang wis suwe.",
    "Emake Sépul, yaiku Bu Kasinem, saben dina dodol jajan neng pasar.",
    "Ki Hajar Dewantara dianggep minangka bapak pendidikan ing Indonesia.",
    "Mas Agus lan Mbak Rini saiki wis manggon neng Purwokerto.",
    "Bupati Banyumas, Pak Achmad, mbuka acara festival budaya Banyumasan.",
    "Pak Guru Ahmad nerangna sejarah Kerajaan Mataram neng kelas.",
    "Presiden Soekarno nate rawuh neng Cilacap nalika taun 1950-an.",
    "Jenderal Sudirman lair neng Purbalingga lan dadi pahlawan nasional Indonesia.",
    "Mas Naryo dadi dhalang wayang kulit sing kondhang neng Banyumas.",
    "Kyai Ahmad mimpin pondok pesantren neng Banjarnegara wis puluhan taun.",
    "Bu Wartini nukokna anake baju anyar neng pasar Ajibarang.",
    "Pak Widodo dilantik dadi camat Sokaraja anyar.",
    "Ki Nartosabdo dadi dhalang kondang saka Jawa Tengah sing ikonik.",
    "Kang Tarno melu lomba nulis basa Banyumasan neng tingkat provinsi.",
    "Prof. Slamet nulis buku bab sejarah Banyumas sing akeh digunakna.",
    "Pak Hasan dadi dokter kondang neng Banyumas wis pirang-pirang taun.",
    "Pak Lurah Kardi mbagi sembako neng warga sing mbutuhna.",
    "Gubernur Jawa Tengah rawuh neng acara pameran batik neng Purwokerto.",
    "Bu Sari dadi perawat setia neng Puskesmas Karanglewas.",
    "Pak Ustadz Fauzi ngajar ngaji saben wengi neng mushola kampung.",
    "Ki Dalang Tarwono nampilna wayang semalam suntuk neng alun-alun.",
    "Mbak Dewi jualan jajan pasar neng pasar Sokaraja saben isuk.",
    "Ibu Kades Sumarni ngatur administrasi désa kanthi becik.",
    "Pak Hadi nyambut gawe minangka supir angkot neng Purwokerto wis taun.",
    "Mbah Marto ngerti akeh sejarah Banyumas saka jaman biyen.",
    "Mas Wahyu ngamen neng pinggir dalan neng Purwokerto wayah sore.",
    "Bu Kepala Sekolah Rina ngumumna libur lebaran kanggo murid-murid.",
    "Kiai Hasyim dadi tokoh agama sing dihormati neng Banyumas.",
    "Dr. Aminah dadi dokter gigi siji-sijine neng kecamatan kono.",
    "Mas Dani dadi juara lomba pidato basa Jawa neng tingkat kabupaten.",
    "Pak Camat Darno miwiti program penghijauan neng kecamatan.",
    "Mas Bejo lan Mbak Parti arep nikah sasi ngarep neng Purwokerto.",
    "Sultan Hamengkubuwono dadi raja Yogyakarta sing dihormati rakyate.",
    "Nyi Rara Kidul dipercaya dadi ratu laut kidul neng kapercayan Jawa.",
    "Prof Bambang kasil dadi guru besar neng Universitas Jenderal Soedirman.",
    # ── LOC ──────────────────────────────────
    "Agus karo Dewi nonton wayang ning Sokaraja bengi.",
    "Dalan menyang Purwokerto ditutup amarga ana longsor gedhe.",
    "Sépul mlaku menyang Sokaraja karo nggawa buku lan tas anyar.",
    "Sépul dolanan neng kali Serayu karo konco-koncone wayah sore.",
    "Neng warung ngarep sekolahan akeh bocah sing tuku jajan.",
    "Kali Serayu mili saka Gunung Slamet nganti tekan segara kidul.",
    "Wong Banyumas akeh sing dodolan neng Pasar Wage Purwokerto.",
    "Wisata Baturraden dadi papan liburan sing akeh dikunjungi neng sasi Lebaran.",
    "Desa Karanglewas duwe kebun teh sing cedhak karo dalan gedhe.",
    "Bocah-bocah dolanan neng alun-alun Banyumas nganti wayah sore.",
    "Gunung Slamet minangka gunung sing paling dhuwur neng Jawa Tengah.",
    "Pantai Logending neng Kebumen dadi papan wisata sing rame neng musim liburan.",
    "Stasiun Purwokerto dadi simpul transportasi penting neng wilayah Banyumas.",
    "Waduk Mrica neng Banjarnegara digunakna kanggo pembangkit listrik.",
    "Pasar Manis Purwokerto rame saben esuk nganti tengah awan.",
    "Kabupaten Kebumen dumunung neng pesisir selatan Jawa Tengah.",
    "Hutan pinus neng Baturraden dadi papan favorit foto-foto bocah enom.",
    "Alun-alun Purbalingga resik lan asri sawise direnovasi.",
    "Gua Jatijajar neng Kebumen dadi objek wisata sejarah sing terkenal.",
    "Pantai Teluk Penyu neng Cilacap asri lan tentrem wayah sore.",
    "Kota Cilacap duwe pelabuhan minyak sing paling gedhe neng Indonesia.",
    "Neng kampung Kauman Purwokerto akeh santri sing ngaji saben wengi.",
    "Kebun Raya Baturraden dadi papan riset tumbuhan neng wilayah Banyumas.",
    "Kampus Unsoed neng Purwokerto gedhe lan akeh mahasiswane.",
    "Kecamatan Ajibarang kawentar karo industri tempe lan tahune.",
    "Jembatan Serayu dadi ikon wisata anyar neng Banyumas.",
    "Désa Kedungreja neng Cilacap kondhang karo ikan asin-e.",
    "Telaga Sunyi neng Baturraden dadi papan wisata alam sing ayem.",
    "Neng jalan raya Wangon akeh warung makan sing dodolan soto.",
    "Pasar Wage Ajibarang dadi pusat ekonomi neng kecamatan.",
    "Terminal Bulupitu dadi pangkalan bis neng Purwokerto.",
    "Gunung Baturagung dadi papan trekking sing populer neng Banyumas.",
    "Hutan neng Baturraden kondhang dadi papan wisata alam sing asri.",
    "Bengawan Solo mili ngliwati Jawa Tengah lan Jawa Timur.",
    "Dalan Jenderal Sudirman dadi dalan utama neng kutha Purwokerto.",
    "Sungai Serayu nglewati pirang-pirang kecamatan neng Banyumas.",
    "Kecamatan Somagede dipasrahna kanggo transmigran saka Jawa wetan biyen.",
    "Rumah Sakit Umum Banyumas dumunung neng tengah-tengah kutha.",
    "Kabupaten Purbalingga kondhang karo industri bulu matane.",
    "Neng pinggir segara Cilacap akeh nelayan sing nggawa jaring.",
    # ── ORG ──────────────────────────────────
    "Pemerintah Banyumas bakal mbangun dalan anyar menyang Ajibarang sesuk.",
    "BNI cabang Banyumas ngadani program menabung kanggo pelajar sekolah.",
    "Madrasah Aliyah Negeri Banyumas nampa siswa anyar taun iki.",
    "Google dadi mesin pencarian sing paling populer digunakna neng Indonesia.",
    "Universitas Jenderal Soedirman dumunung neng Purwokerto, Banyumas.",
    "TNI lan Polri melu njaga kamanan neng acara peringatan kemerdekaan.",
    "SD Negeri Sokaraja 1 nampa bantuan buku saka Dinas Pendidikan.",
    "Muhammadiyah mbangun sekolah anyar neng pinggir kutha Cilacap.",
    "PLN Banyumas nambah jaringan listrik anyar kanggo warga kabupaten.",
    "NU lan Muhammadiyah bebarengan ngadani bakti sosial neng Banjarnegara.",
    "BRI cabang Purwokerto mbuka layanan tabungan anyar kanggo petani.",
    "Polri Banyumas ngadani razia knalpot brong neng dalan utama kabupaten.",
    "Dinas Pertanian Banyumas menehi bibit padi gratis kanggo petani.",
    "Pemerintah Cilacap rapat bahas anggaran pembangunan infrastruktur daerah.",
    "SMA Negeri 1 Purwokerto kasil ngantarake siswane menyang PTN.",
    "Lembaga Kesehatan Sokaraja ngadani vaksinasi massal kanggo warga.",
    "Polri Jawa Tengah ngirimna bantuan logistik neng korban bencana alam.",
    "Dinas Pendidikan Banyumas ngumumna jadwal penerimaan siswa anyar.",
    "Koperasi Nelayan Cilacap mbantu anggotane dodol iwak hasil tangkapan.",
    "PLN cabang Purwokerto ngumumna jadwal perawatan jaringan listrik.",
    "Kantor Pos Purwokerto dadi papan pengiriman paket paling rame.",
    "SMK Negeri 1 Banyumas duwe jurusan teknik otomotif sing kondhang.",
    "Pemerintah Banyumas ngluarake status siaga bencana amarga udan lebat.",
    "Koperasi Tani Makmur neng Ajibarang nampung hasil panen petani.",
    "Yayasan Al Irsyad mbangun madrasah anyar neng Cilacap.",
    "Muhammadiyah Banyumas ngadani pelatihan kurikulum anyar kanggo guru-guru.",
    "TNI AU duwe pangkalan neng Cilacap sing njaga wilayah udara kidul Jawa.",
    "Ikatan Dokter Indonesia cabang Banyumas ngadani seminar kesehatan.",
    "Lembaga Seni Banyumas nampilna pertunjukan lengger neng festival budaya.",
    "Pemerintah Kabupaten Cilacap ngadani lomba inovasi desa taun iki.",
    "BRI Banyumas ngadani program kredit usaha rakyat kanggo pelaku UMKM.",
    "Kantor Kecamatan Purwojati renovasi gedung kanggo layanan publik.",
    "Polri Sokaraja nangkep copet sing operasi neng pasar Kliwon.",
    "BRI Unit Ajibarang nyedhiakna kredit murah kanggo pengusaha cilik.",
    "Wikipedia dadi sumber referensi sing akeh digunakna pelajar.",
    "Koperasi Dagang Banyumas mbantu UKM ekspor produk lokal.",
    "Madrasah Ibtidaiyah Negeri Purbalingga nampa dana bantuan operasional.",
    "Lembaga Kepanduan Banyumas ngadani jambore tingkat kabupaten.",
    "Pemerintah Purbalingga bakal ngadani festival bulu mata internasional.",
    "NU lan Muhammadiyah bebarengan ngajak donor darah rutin neng Banyumas.",
    # ── TIME ─────────────────────────────────
    "Dina Senen, tanggal 17 Agustus 1945, Indonesia merdeka saka penjajah.",
    "Taun 2023 akeh wong Banyumas sing mulai nulis neng platform digital.",
    "Wiwit sasi Januari nganti Maret, udan teka saben dina tanpa lèrèn.",
    "Wingi esuk, sekitar jam pitu, Sépul mangkat sekolah karo batire.",
    "Neng taun 1998, akeh kedadeyan gedhe sing ora bisa dilalèkna.",
    "Rebo Kliwon dianggep dina kang wingit neng tradhisi Jawa Banyumasan.",
    "Saben wulan Ramadhan, wong-wong padha tarawih neng masjid bebarengan.",
    "Acara wisuda bakal dianakna tanggal 20 Desember taun iki neng GOR.",
    "Wiwit jam songo nganti jam rolas, pasar pancen rame banget.",
    "Neng jaman Belanda, akeh wong Banyumas sing kerja rodi nggawe dalan.",
    "Saben Rebo Wage, Mbah Citro teka neng pasar kanggo dodolan tempe.",
    "Taun 1965 dadi salah sijine taun paling kelam neng sejarah Indonesia.",
    "Mulai tanggal siji Januari, kabeh harga resmi naik neng seluruh Indonesia.",
    "Jam loro bengi, Sépul krungu swara aneh saka mburi omah.",
    "Neng sasi Syawal, akeh keluarga sing padha sungkeman lan silaturahmi.",
    "Pesta rakyat bakal dianakna tanggal 17 Agustus neng alun-alun Purwokerto.",
    "Saben Minggu esuk, warga padha senam bareng neng lapangan.",
    "Neng taun 2020, pandemi Covid-19 nggawe akeh sekolah ditutup.",
    "Wayah jam enem sore, adzan Maghrib saka masjid-masjid muni bebarengan.",
    "Proses panen pari biasane ditindakna neng sasi April utawa Mei.",
    "Neng tanggal 28 Oktober, kabeh murid melu upacara Sumpah Pemuda.",
    "Acara pengajian rutin diadakna saben Jemuwah malem neng langgar.",
    "Wiwit jam lima esuk, pedagang pasar wis rame dodolan.",
    "Sasi Rajab dadi wulan sing diistimewake neng tradhisi Islam.",
    "Neng taun 2010, Gunung Merapi mbledhos lan nggawe akeh korban.",
    "Saben taun neng sasi Desember, sekolah ngadani pentas seni siswa.",
    "Jam setengah pitu bengi, Emak wis masak kanggo makan wengi.",
    "Tanggal 21 April diperingati minangka Hari Kartini neng Indonesia.",
    "Wiwit jaman semono, tradhisi nyadran wis dilakoni dening wong Banyumasan.",
    "Saben Lebaran neng wulan Syawal, omah Pak Citro rame dikunjungi sedulur.",
    "Neng tahun ajaran anyar, akeh siswa anyar sing mlebu SMP.",
    "Jam sewelas siang, pelajaran meh rampung lan bocah-bocah wis kangen mangan.",
    "Neng wulan Maret taun 2020, pemerintah ngumumna status pandemi.",
    "Saben bengi Jumat, pemuda desa nganakna kegiatan yasinan bebarengan.",
    "Neng taun 1945, proklamasi kemerdekaan Indonesia dibacakna neng Jakarta.",
    "Tanggal 2 Mei saben taun diperingati minangka Hari Pendidikan Nasional.",
    "Wiwit sasi Oktober, musim kemarau wis mulai watara neng Jawa.",
    "Jam telu esuk, Pak Sabar tangi kanggo sembahyang tahajud.",
    "Neng abad ping wolulas, Banyumas wis dadi wilayah administratif Mataram.",
    "Saben Selasa Kliwon, pasar malem diadakna neng lapangan desa.",
    # ── MISC ─────────────────────────────────
    "Emak nukokna Sépul tas anyar lan buku kanggo sinau neng sekolah.",
    "Sépul golek welut neng blumbang karo nggawa ember lan jala cilik.",
    "Sépul nukokna Emak buku masak anyar saka toko neng pasar Sokaraja.",
    "Neng pasar Sokaraja ana dodolan iwak, welut, sayuran, lan jajan pasar.",
    "Neng pasar Cilacap ana dodolan iwak segara lan welut goreng sing rame.",
    "Pak Joko nukokna Sépul tas sekolah anyar karo buku pelajaran.",
    "Mas Budi golek welut neng kali karo nggawa ember lan jala gedhe.",
    "Emak masak welut goreng bumbu bawang kanggo mangan bengi sekeluarga.",
    "Neng pasar Ajibarang, jajan pasar lan iwak bakar rame dikunjungi warga.",
    "Sépul nggawa ember cilik lan jala kanggo golek welut neng blumbang.",
    "Pak Hamidin nukokna anake buku cerita lan tas anyar kanggo sekolah.",
    "Neng pasar Purwokerto, dodolan iwak segar lan jajan pasar akeh pilihannya.",
    "Bu Wartini tuku tas anyar lan buku resep masak neng toko Sokaraja.",
    "Mas Tarno nggawa jala lan ember menyang kali Serayu kanggo golek iwak.",
    "Sépul nukokna buku gambar anyar kanggo prakarya neng sekolah.",
    "Emak tuku ember anyar neng toko kanggo golek welut neng blumbang.",
    "Pak Kardi golek welut neng blumbang karo anake nggawa jala cilik.",
    "Mbak Dewi nukokna adike tas anyar lan buku tulis neng toko Purwokerto.",
    "Sépul golek iwak neng kali karo nggawa ember lan jala saka omah.",
    "Pak Lurah nukokna buku pelajaran anyar lan tas kanggo murid anyar.",
    "Neng pasar Wangon, welut goreng lan iwak pindang dadi dodolan paling laris.",
    "Emak nukokna Sépul tas anyar warna biru karo buku catetan kanggo sekolah.",
    "Mas Bejo golek welut lan iwak neng blumbang karo nggawa ember gedhe.",
    "Sépul nggawa jala anyar lan ember kanggo golek welut neng sawah mburi omah.",
    "Pak Tarno nukokna anake buku sejarah lan tas anyar neng pasar Purwokerto.",
    "Sépul golek buku bab sains neng perpustakaan sekolah kanggo tugas.",
    "Kaleng susu bekas dimanfaatna kanggo pot kembang neng teras omah.",
    "Sate klathak Banyumas dibakar nganggo arang kayu supaya luwih gurih.",
    "Pak Budi nukokna anake tas anyar lan buku tulis kanggo sekolah.",
    "Sépul lan Budi golek iwak neng kali karo nggawa ember lan jala.",
    "Buku pelajaran Bahasa Jawa neng kelas VI duwe materi bab aksara Jawa.",
    "Emak nukokna Sépul buku pelajaran anyar lan tas warna ireng.",
    "Mas Kardi nukokna anake tas sekolah anyar lan buku paket agama.",
    "Neng blumbang ngarep omah, Sépul golek welut karo nggawa ember.",
    "Emak masak welut goreng lan iwak bumbu kuning kanggo mangan sekeluarga.",
    "Neng warung bubur Purwokerto, dodolan jajan lan iwak pindang rame.",
    "Sépul nukokna tas anyar lan buku gambar kanggo prakarya neng sekolah.",
    "Mas Tarno golek welut lan iwak neng kali karo nggawa jala cilik.",
    "Pak Guru nukokna buku paket lan tas kanggo murid kelas siji.",
    "Neng pasar Banyumas, jajan tradisional lan welut goreng rame didol.",
    "Sépul tuku tas anyar lan buku neng toko karo nggawa ember cilik.",
    "Emak nukokna Sépul tas anyar lan buku kanggo sinau neng Ajibarang.",
    "Neng kali mburi omah, Sépul golek iwak karo nggawa ember lan jala.",
]


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">Kelompok Ngapak Keren · NLP Bahasa Daerah</div>
  <h1><span>BanyuNer</span></h1>
  <p>Named Entity Recognition untuk Bahasa Banyumasan — mengidentifikasi entitas Person, Location, Organization, Miscellaneous, dan Time secara otomatis dari teks bahasa daerah.</p>
  <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem; padding:0.8rem 1.2rem; background:rgba(59,31,14,0.3); border-radius:10px; border:1px solid rgba(200,146,42,0.15);">
<span class="legend-item"><span class="legend-dot" style="background:#C0552A;"></span> Person (PER)</span>
<span class="legend-item"><span class="legend-dot" style="background:#3A9090;"></span> Location (LOC)</span>
<span class="legend-item"><span class="legend-dot" style="background:#B07040;"></span> Organization (ORG)</span>
<span class="legend-item"><span class="legend-dot" style="background:#9060C0;"></span> Miscellaneous (MISC)</span>
<span class="legend-item"><span class="legend-dot" style="background:#5E9A50;"></span> Time (TIME)</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────
col_in, col_out = st.columns([1, 1.3], gap="large")

# ── INPUT PANEL ──────────────────────────────
with col_in:
    st.markdown('<div class="section-title">Pilih Kalimat Contoh</div>', unsafe_allow_html=True)

    sample_choice = st.selectbox(
        "Pilih kalimat:",
        ["— Pilih contoh kalimat —"] + SAMPLES,
        label_visibility="collapsed",
    )
    user_text = "" if sample_choice.startswith("—") else sample_choice

    if user_text:
        st.markdown(
            f'<div class="card" style="font-size:0.95rem; line-height:1.8; margin-top:0.4rem;">{user_text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)
    run_btn = st.button("PROSES  /  PREDICT")

    # Model information panel
    st.markdown("""
    <div class="card" style="margin-top:1.2rem; font-size:0.83rem; color:#C9B99A; line-height:1.85;">
    <b style="color:var(--gold); display:block; margin-bottom:0.5rem;">Informasi Model</b>
    <div style="display:flex; justify-content:space-between;">
      <span>Model</span><span style="color:var(--cream); font-family:'Source Code Pro',monospace;">CRF (L-BFGS)</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
      <span>Dataset</span><span style="color:var(--cream); font-family:'Source Code Pro',monospace;">870 kalimat</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
      <span>Label</span><span style="color:var(--cream); font-family:'Source Code Pro',monospace;">PER · LOC · ORG · TIME · MISC</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Evaluation summary panel
    st.markdown("""
    <div class="card" style="margin-top:0.6rem; font-size:0.83rem; color:#C9B99A;">
    <b style="color:var(--gold); display:block; margin-bottom:0.8rem;">Evaluasi Model &mdash; F1-score (Test Set)</b>
    <div style="margin-bottom:9px;">
      <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px;">
        <span style="color:#F5A07A; font-family:'Source Code Pro',monospace;">PER</span><span>0.96</span>
      </div>
      <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:96%; background:#C0552A;"></div></div>
    </div>
    <div style="margin-bottom:9px;">
      <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px;">
        <span style="color:#A3D98E; font-family:'Source Code Pro',monospace;">TIME</span><span>0.89</span>
      </div>
      <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:89%; background:#5E9A50;"></div></div>
    </div>
    <div style="margin-bottom:9px;">
      <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px;">
        <span style="color:#C8A0F0; font-family:'Source Code Pro',monospace;">MISC</span><span>0.56</span>
      </div>
      <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:56%; background:#9060C0;"></div></div>
    </div>
    <div style="margin-bottom:9px;">
      <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px;">
        <span style="color:#7ADADA; font-family:'Source Code Pro',monospace;">LOC</span><span>0.40</span>
      </div>
      <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:40%; background:#3A9090;"></div></div>
    </div>
    <div style="margin-bottom:4px;">
      <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px;">
        <span style="color:#E0B888; font-family:'Source Code Pro',monospace;">ORG</span><span>0.00</span>
      </div>
      <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:0%; background:#B07040;"></div></div>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ── OUTPUT PANEL ─────────────────────────────
with col_out:
    st.markdown('<div class="section-title">Hasil Prediksi</div>', unsafe_allow_html=True)

    if run_btn:
        if not user_text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Memproses teks…"):
                time.sleep(0.6)   # mimic model latency
                results = predict_ner(user_text)

            entities = [r for r in results if r["label"] != "O"]
            entity_counts = Counter(r["label"] for r in entities)
            avg_conf = (
                sum(r["confidence"] for r in entities) / len(entities)
                if entities else 0.0
            )

            # ── metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Token", len(results))
            m2.metric("Entitas", len(entities))
            m3.metric("Avg Confidence", f"{avg_conf:.1%}")

            translated_text, translated_count = translate_banyumasan(user_text)

            tab1, tab2, tab3, tab4 = st.tabs(["Highlight", "Tabel", "Statistik", "Terjemahan"])

            with tab1:
                html_out = build_highlighted_html(results)
                st.markdown(
                    f'<div class="card" style="line-height:2.2; font-size:1rem;">{html_out}</div>',
                    unsafe_allow_html=True,
                )

            with tab2:
                if entities:
                    css_map   = {"PER":"tag-PER","LOC":"tag-LOC","ORG":"tag-ORG","MISC":"tag-MISC","TIME":"tag-TIME"}
                    color_map = {"PER":"#C0552A","LOC":"#3A9090","ORG":"#B07040","MISC":"#9060C0","TIME":"#5E9A50"}
                    rows = ""
                    for r in entities:
                        lbl   = r["label"]
                        conf  = r["confidence"]
                        css   = css_map.get(lbl, "")
                        color = color_map.get(lbl, "#888")
                        rows += f"""
                        <tr>
                          <td><b>{r['token']}</b></td>
                          <td><span class="entity-highlight {css}" style="font-size:0.8rem;">{lbl}</span></td>
                          <td>
                            <div style="display:flex; align-items:center; gap:8px;">
                              <span style="font-family:'Source Code Pro',monospace; min-width:50px;">{conf:.2%}</span>
                              <div class="conf-bar-wrap" style="flex:1; height:6px;">
                                <div class="conf-bar-fill" style="width:{conf*100:.1f}%; background:{color};"></div>
                              </div>
                            </div>
                          </td>
                        </tr>"""
                    st.markdown(
                        f'<table class="result-table"><thead><tr><th>Token</th><th>Label</th><th>Confidence</th></tr></thead><tbody>{rows}</tbody></table>',
                        unsafe_allow_html=True,
                    )

                    # download
                    csv = pd.DataFrame(entities).to_csv(index=False).encode()
                    st.download_button(
                        "Download CSV",
                        data=csv,
                        file_name="ner_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info("Tidak ada entitas terdeteksi dalam teks ini.")

            with tab3:
                if entities:
                    st.markdown("**Distribusi Entitas**")
                    for lbl, cnt in entity_counts.items():
                        pct = cnt / len(entities)
                        color_map = {"PER":"#C0552A","LOC":"#3A9090","ORG":"#B07040","MISC":"#9060C0","TIME":"#5E9A50"}
                        color = color_map.get(lbl, "#888")
                        st.markdown(f"""
                        <div style="margin-bottom:10px;">
                          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
                            <span style="color:{color}; font-family:'Source Code Pro',monospace;">{lbl}</span>
                            <span>{cnt} token ({pct:.0%})</span>
                          </div>
                          <div class="conf-bar-wrap">
                            <div class="conf-bar-fill" style="width:{pct*100:.1f}%; background:{color};"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("**Confidence per Entitas**")
                    for r in entities:
                        lbl = r["label"]
                        color_map = {"PER":"#C0552A","LOC":"#3A9090","ORG":"#B07040","MISC":"#9060C0","TIME":"#5E9A50"}
                        color = color_map.get(lbl, "#888")
                        st.markdown(f"""
                        <div style="margin-bottom:8px;">
                          <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:3px;">
                            <span><b>{r['token']}</b> <span style="color:{color}; font-family:'Source Code Pro',monospace; font-size:0.7rem;">{lbl}</span></span>
                            <span style="color:{color};">{r['confidence']:.2%}</span>
                          </div>
                          <div class="conf-bar-wrap">
                            <div class="conf-bar-fill" style="width:{r['confidence']*100:.1f}%; background:{color};"></div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Tidak ada entitas untuk ditampilkan.")

            with tab4:
                total_tokens = len([t for t in re.findall(r"[\w']+", user_text)])
                coverage = translated_count / total_tokens if total_tokens else 0
                c1, c2 = st.columns(2)
                c1.metric("Kata Diterjemahkan", translated_count)
                c2.metric("Cakupan Kamus", f"{coverage:.0%}")
                st.markdown(
                    '<div class="section-title" style="margin-top:1rem;">Teks Asli (Banyumasan)</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="card" style="font-size:1rem; line-height:1.8;">{user_text}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-title" style="margin-top:1rem;">Terjemahan (Bahasa Indonesia)</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="card" style="font-size:1rem; line-height:1.8; border-color:rgba(200,146,42,0.55);">{translated_text}</div>',
                    unsafe_allow_html=True,
                )
                if coverage < 0.5:
                    st.info(
                        "Beberapa kata tidak ditemukan di kamus. "
                        "Kata-kata tersebut ditampilkan apa adanya."
                    )

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3.5rem 2rem; color:#7A5A3A;">
          <div style="font-size:3rem; margin-bottom:1rem;"></div>
          <div style="font-family:'Playfair Display',serif; font-size:1.1rem; color:#A07050;">
            Masukkan teks Banyumasan<br>lalu klik <b style="color:var(--gold);">PROSES</b> untuk melihat hasil NER.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""<hr>""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#5A3A20; font-size:0.8rem; padding:0.5rem 0 1.5rem; font-family:'Source Code Pro',monospace;">
  BanyuNer · Kelompok Ngapak Keren · Named Entity Recognition · Bahasa Banyumasan
</div>
""", unsafe_allow_html=True)
