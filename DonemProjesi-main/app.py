"""
app.py — AI Çağrı Merkezi Analiz Platformu

Modüller:
  - Metin Analizi : BERT tabanlı duygu tahmini
  - Çağrı Analizi : Whisper STT + Pyannote diarization + Multimodal fusion
  - Canlı Kayıt   : Stereo kayıt (Mikrofon + WhatsApp)
"""

import os
import torch
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

from src.text_analyzer  import TextAnalyzer
from src.audio_analyzer import ProfessionalAudioAnalyzer
from src.stt_engine     import SpeechToTextEngine
from src.fusion_engine  import MultimodalFusion
from src.call_analyzer  import CallAnalyticEngine
from recorder_page      import render_recorder_page

# ── SABITLER ─────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bert_gercekveri")
TEMP_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "temp_call_audio.wav")

SENTIMENT_COLORS = {
    "OLUMLU":  ("#d4edda", "#28a745"),
    "OLUMSUZ": ("#f8d7da", "#dc3545"),
    "KİNAYE":  ("#f8d7da", "#dc3545"),
    "NEGATİF": ("#f8d7da", "#dc3545"),
    "POZİTİF": ("#d4edda", "#28a745"),
    "NOTR":    ("#e8f0fe", "#1a73e8"),
    "DEFAULT": ("#e2e3e5", "#6c757d"),
}

# ── BERT TAHMİN SINIFI ────────────────────────────────────────────────────────
class SentimentPredictor:
    LABELS = {0: "NEGATİF", 1: "NÖTR", 2: "POZİTİF"}

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return "BOŞ", 0.0
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        score, idx = torch.max(probs, dim=1)
        return self.LABELS.get(idx.item(), "BİLİNMİYOR"), score.item()


# ── MODEL YÜKLEME ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_engines():
    predictor = SentimentPredictor()
    text_e    = TextAnalyzer()
    audio_e   = ProfessionalAudioAnalyzer()
    stt_e     = SpeechToTextEngine()
    fusion_e  = MultimodalFusion()
    call_ai   = CallAnalyticEngine(HF_TOKEN, text_e, audio_e, stt_e, fusion_e)
    return predictor, call_ai


# ── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────────
def sentiment_colors(label: str) -> tuple[str, str]:
    label_upper = label.strip().upper()
    for key, colors in SENTIMENT_COLORS.items():
        if key in label_upper:
            return colors
    return SENTIMENT_COLORS["DEFAULT"]


def render_segment(entry: dict):
    role = entry.get("role", "")
    sentiment = entry.get("sentiment", "-")

    if role == "TEMSİLCİ":
        bg, border = "#e2e3e5", "#6c757d"
    else:
        bg, border = sentiment_colors(sentiment)

    sentiment_info = f" | <b>Duygu: {sentiment}</b>" if role == "MÜŞTERİ" else ""
    st.markdown(f"""
    <div style="background:{bg};color:#383d41;padding:15px;border-radius:12px;
                margin:10px 0;border-left:8px solid {border};
                box-shadow:1px 1px 3px rgba(0,0,0,0.1);">
        <div style="margin-bottom:5px;">
            <strong>[{entry['time']}] {entry['role']}</strong>{sentiment_info}
        </div>
        <div style="font-size:1.05em;line-height:1.4;">{entry['text']}</div>
    </div>
    """, unsafe_allow_html=True)


# ── SAYFA AYARLARI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Çağrı Analiz Merkezi", layout="wide", page_icon="🎙️")
st.markdown("""
<style>
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
}
</style>
""", unsafe_allow_html=True)

# ── MOTORLARI BAŞLAT ─────────────────────────────────────────────────────────
with st.spinner("🤖 Yapay zeka motorları başlatılıyor..."):
    bert_predictor, call_center_ai = load_all_engines()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4233/4233830.png", width=100)
st.sidebar.title("Analiz Menüsü")
app_mode = st.sidebar.radio(
    "Modül Seçin:",
    ["🏠 Ana Sayfa", "📝 Metin Analizi", "📞 Çağrı Analizi", "🎤 Ses Analizi", "🎙️ Canlı Kayıt"]
)
st.sidebar.markdown("---")
st.sidebar.info("Geliştirici: Uğur Arslan, Fuat Kuşkonmaz, Melih Can")

# ── SAYFALAR ──────────────────────────────────────────────────────────────────
if app_mode == "🏠 Ana Sayfa":
    st.title("🎙️ AI Çağrı Merkezi Analiz Platformu")
    st.markdown("""
    Bu platform iki ana modüle sahiptir:

    - **Metin Analizi:** Eğitilmiş BERT modeliyle metinden duygu tahmini yapar.
    - **Çağrı Analizi:** Ses kaydını metne döker, konuşmacıları ayırır ve
      metin + ses tonu birleştirilerek multimodal duygu analizi yapar.
    - **Canlı Kayıt:** Mikrofon ve WhatsApp sesini ayrı kanallara kaydeder.
    """)

elif app_mode == "📝 Metin Analizi":
    st.header("🔍 Metin Duygu Analizi")
    user_input = st.text_input(
        "Analiz edilecek metni yazın:",
        placeholder="Hizmetinizden çok memnun kaldım..."
    )
    if st.button("Analiz Et"):
        if user_input:
            label, confidence = bert_predictor.predict(user_input)
            col1, col2 = st.columns(2)
            col1.metric("Tahmin", label)
            col2.metric("Güven", f"%{confidence * 100:.1f}")
            st.progress(confidence)
        else:
            st.warning("Lütfen bir metin girin.")

elif app_mode == "📞 Çağrı Analizi":
    st.header("🎙️ Multimodal Çağrı Analizi")

    # Canlı kayıttan gelen dosya otomatik yükle
    pending = st.session_state.get("pending_audio")
    if pending and os.path.exists(pending):
        st.info(f"Canlı kayıttan yüklendi: `{os.path.basename(pending)}`")
        audio_path = pending
        st.audio(pending)
        st.session_state["pending_audio"] = None
    else:
        uploaded_file = st.file_uploader("Çağrı kaydı yükleyin (.wav)", type=["wav"])
        audio_path = None
        if uploaded_file:
            with open(TEMP_AUDIO, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.audio(uploaded_file)
            audio_path = TEMP_AUDIO

    if audio_path and st.button("Çağrıyı Analiz Et"):
        with st.status("Analiz ediliyor...", expanded=True) as status:
            st.write("🤖 Ses metne çevriliyor, duygu analizi yapılıyor...")
            report = call_center_ai.analyze_full_audio(audio_path)
            status.update(label="Analiz Tamamlandı!", state="complete")

        st.subheader("📋 Analiz Raporu")
        for entry in report:
            render_segment(entry)

elif app_mode == "🎤 Ses Analizi":
    st.header("🎤 Tek Ses Kaydı Analizi")
    st.markdown("Tek kişiye ait bir ses kaydı yükleyin. Ses tonu + metin birleştirilerek duygu analizi yapılır.")

    uploaded_file = st.file_uploader("Ses dosyası yükleyin (.wav)", type=["wav"], key="single_audio")

    if uploaded_file:
        with open(TEMP_AUDIO, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(uploaded_file)

        if st.button("Analiz Et"):
            with st.spinner("🤖 Analiz yapılıyor..."):
                text      = call_center_ai.stt_engine.transcribe(TEMP_AUDIO)
                text_res  = call_center_ai.text_engine.analyze(text)
                audio_res = call_center_ai.audio_engine.analyze(TEMP_AUDIO)
                fusion    = call_center_ai.fusion_engine.merge(text_res, audio_res)

            st.markdown("### 💬 Tespit Edilen Metin")
            st.info(text)

            col1, col2, col3 = st.columns(3)
            col1.metric("📝 Metin (BERT)",   text_res['label'].upper())
            col2.metric("🎤 Ses Tonu (SVM)", audio_res['label'].upper())
            col3.metric("✅ Sonuç",            fusion['final_sentiment'].upper())

            if fusion['sarcasm_detected']:
                st.warning("⚠️ Kinaye tespit edildi! Metin ile ses tonu çelişiyor.")

            with st.expander("🔍 Detaylı Skor"):
                st.write(f"**Metin güveni:** %{text_res['score']*100:.1f}")
                st.write(f"**Ses güveni:** %{audio_res['confidence']*100:.1f}")
                st.write(f"**Fusion olasılıkları:** Olumsuz={fusion['fusion_probs'][0]:.2f} | Nötr={fusion['fusion_probs'][1]:.2f} | Olumlu={fusion['fusion_probs'][2]:.2f}")

elif app_mode == "🎙️ Canlı Kayıt":
    render_recorder_page(call_center_ai, render_segment)
