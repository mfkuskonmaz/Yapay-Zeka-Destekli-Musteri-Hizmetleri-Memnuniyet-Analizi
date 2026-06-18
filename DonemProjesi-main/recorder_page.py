import os
import datetime
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st
import librosa

SAMPLE_RATE = 16000
MIC_ID      = 1   # Microphone Array (AMD Audio Device)
VIRTUAL_ID  = 8   # Voicemeeter Out B1 (WhatsApp sesi)
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "recordings")
TEMP_MONO   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp", "temp_recorder_mono.wav")
os.makedirs(SAVE_DIR, exist_ok=True)


def _record(duration_seconds: int, result: dict):
    mic_data     = []
    virtual_data = []

    mic_sr     = int(sd.query_devices(MIC_ID)['default_samplerate'])
    virtual_sr = int(sd.query_devices(VIRTUAL_ID)['default_samplerate'])

    def mic_cb(indata, frames, time, status):
        mic_data.append(indata[:, 0].copy())

    def virtual_cb(indata, frames, time, status):
        virtual_data.append(indata[:, 0].copy())

    stop_event = result.get("stop_event")

    with sd.InputStream(device=MIC_ID,     samplerate=mic_sr,
                        channels=1, callback=mic_cb), \
         sd.InputStream(device=VIRTUAL_ID, samplerate=virtual_sr,
                        channels=1, callback=virtual_cb):
        stop_event.wait(timeout=duration_seconds)

    left  = np.concatenate(mic_data)     if mic_data     else np.array([])
    right = np.concatenate(virtual_data) if virtual_data else np.array([])

    if mic_sr != SAMPLE_RATE and len(left) > 0:
        left  = librosa.resample(left,  orig_sr=mic_sr,     target_sr=SAMPLE_RATE)
    if virtual_sr != SAMPLE_RATE and len(right) > 0:
        right = librosa.resample(right, orig_sr=virtual_sr, target_sr=SAMPLE_RATE)

    min_len = min(len(left), len(right))
    if min_len == 0:
        result["error"] = "Ses verisi alınamadı."
        return

    stereo   = np.stack([left[:min_len], right[:min_len]], axis=1)
    filename = datetime.datetime.now().strftime("cagri_%Y%m%d_%H%M%S.wav")
    filepath = os.path.join(SAVE_DIR, filename)
    sf.write(filepath, stereo, SAMPLE_RATE)
    result["filepath"] = filepath
    result["filename"] = filename


def _analyze(fpath: str, call_center_ai):
    """Stereo WAV'ı mono'ya çevirip ses analizi yapar."""
    audio, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
    sf.write(TEMP_MONO, audio, SAMPLE_RATE)

    text      = call_center_ai.stt_engine.transcribe(TEMP_MONO)
    text_res  = call_center_ai.text_engine.analyze(text)
    audio_res = call_center_ai.audio_engine.analyze(TEMP_MONO)
    fusion    = call_center_ai.fusion_engine.merge(text_res, audio_res)
    return text, text_res, audio_res, fusion


def _show_analysis(fpath: str, call_center_ai):
    with st.spinner("🤖 Analiz yapılıyor..."):
        text, text_res, audio_res, fusion = _analyze(fpath, call_center_ai)

    st.markdown("### 💬 Tespit Edilen Metin")
    st.info(text)

    col1, col2, col3 = st.columns(3)
    col1.metric("📝 Metin (BERT)",   text_res['label'].upper())
    col2.metric("🎤 Ses Tonu (SVM)", audio_res['label'].upper())
    col3.metric("✅ Sonuç",           fusion['final_sentiment'].upper())

    if fusion['sarcasm_detected']:
        st.warning("⚠️ Kinaye tespit edildi! Metin ile ses tonu çelişiyor.")

    with st.expander("🔍 Detaylı Skor"):
        st.write(f"**Metin güveni:** %{text_res['score']*100:.1f}")
        st.write(f"**Ses güveni:** %{audio_res['confidence']*100:.1f}")
        st.write(f"**Fusion olasılıkları:** Olumsuz={fusion['fusion_probs'][0]:.2f} | Nötr={fusion['fusion_probs'][1]:.2f} | Olumlu={fusion['fusion_probs'][2]:.2f}")


def render_recorder_page(call_center_ai, render_segment):
    st.header("🎙️ Canlı Çağrı Kaydı")

    for key, val in [("recording", False), ("stop_event", None),
                     ("record_result", {}), ("thread", None)]:
        if key not in st.session_state:
            st.session_state[key] = val

    duration = st.slider("Maksimum kayıt süresi (saniye)", 30, 600, 120, 30)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔴 Kaydı Başlat", disabled=st.session_state.recording):
            stop_event = threading.Event()
            result     = {"stop_event": stop_event}
            thread     = threading.Thread(target=_record, args=(duration, result), daemon=True)
            st.session_state.recording     = True
            st.session_state.stop_event    = stop_event
            st.session_state.record_result = result
            st.session_state.thread        = thread
            thread.start()
            st.rerun()

    with col2:
        if st.button("⏹️ Kaydı Durdur", disabled=not st.session_state.recording):
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            if st.session_state.thread:
                st.session_state.thread.join(timeout=5)
            st.session_state.recording = False
            st.rerun()

    if st.session_state.recording:
        st.success("🔴 Kayıt devam ediyor...")
    else:
        result = st.session_state.record_result
        if result.get("filepath"):
            st.success(f"✅ Kaydedildi: `{result['filename']}`")
            st.audio(result["filepath"])
            if st.button("📞 Analiz Et"):
                _show_analysis(result["filepath"], call_center_ai)
        elif result.get("error"):
            st.error(f"Hata: {result['error']}")

    # Geçmiş kayıtlar
    st.markdown("---")
    st.subheader("📁 Kaydedilen Çağrılar")
    files = sorted(
        [f for f in os.listdir(SAVE_DIR) if f.endswith(".wav")],
        reverse=True
    )
    if files:
        for f in files[:10]:
            fpath = os.path.join(SAVE_DIR, f)
            cols  = st.columns([3, 1, 1])
            cols[0].markdown(f"🎵 `{f}`")
            with open(fpath, "rb") as fp:
                cols[1].download_button("⬇️ İndir", fp, file_name=f, key=f"dl_{f}")
            if cols[2].button("📞 Analiz Et", key=f"analyze_{f}"):
                _show_analysis(fpath, call_center_ai)
    else:
        st.info("Henüz kayıt yok.")
