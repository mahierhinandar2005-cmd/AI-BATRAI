import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Battery Assistant", page_icon="🔋", layout="wide")

# ==================== CUSTOM CSS RINGAN ====================
st.markdown("""
<style>
    .stApp { background: #0D0D0D; }
    .chat-container { max-width: 900px; margin: 0 auto; padding: 1rem; }
    .bot-avatar {
        width: 45px; height: 45px;
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; flex-shrink: 0;
    }
    .bot-message-wrapper { display: flex; gap: 12px; margin-bottom: 24px; }
    .bot-bubble {
        background: #1E1E1E; border-radius: 20px 20px 20px 4px;
        padding: 16px 20px; color: #E0E0E0; border: 1px solid #2E2E2E;
        max-width: 80%;
    }
    .user-bubble {
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border-radius: 20px 20px 4px 20px; padding: 12px 20px;
        color: white; max-width: 70%; margin-left: auto; margin-bottom: 24px;
    }
    .info-btn { background: rgba(16,185,129,0.15); border: 1px solid #10B981; border-radius: 30px; padding: 4px 12px; color: #10B981; font-size: 0.7rem; cursor: pointer; display: inline-block; }
    .footer { text-align: center; color: #666; font-size: 0.7rem; padding: 20px; margin-top: 80px; }
    .stPopover { background: #1E1E1E !important; border: 1px solid #10B981 !important; border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    model = joblib.load('model_soh.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

try:
    model, scaler_X, scaler_y = load_model()
except:
    st.error("Model tidak ditemukan.")
    st.stop()

# ==================== PARAMETER INFO ====================
PARAM_INFO = {
    "cycle": {"name": "🔄 Aging Cycle", "detail": "Jumlah siklus charge-discharge. Semakin tinggi, baterai semakin aus.", "range": "0-2000", "good": "<300", "bad": ">600", "source": "BMS / alat diagnostik"},
    "soc": {"name": "🔋 SOC (%)", "detail": "Level pengisian baterai. Ideal 20-80%.", "range": "0-100", "good": "20-80%", "bad": "<20% atau >80%", "source": "Dashboard mobil"},
    "rint": {"name": "⚡ R_int (%)", "detail": "Internal resistance. Semakin tinggi, semakin rusak.", "range": "0-200", "good": "≤110%", "bad": ">150%", "source": "Alat diagnostik"},
    "ocv": {"name": "🔌 OCV (V)", "detail": "Tegangan diam baterai. Sehat di atas 3.8V.", "range": "3.0-4.5", "good": "≥3.9V", "bad": "<3.7V", "source": "Voltmeter/BMS"},
    "freq": {"name": "📊 Frequency (Hz)", "detail": "Frekuensi pengukuran EIS. Parameter teknis.", "range": "0.1-10000", "good": None, "bad": None, "source": "Alat EIS", "note": "Parameter pengukuran"},
    "zmod": {"name": "📈 Zmod (Ohm)", "detail": "Impedansi total. Meningkat saat degradasi.", "range": "0.005-0.05", "good": "Rendah", "bad": "Tinggi", "source": "Alat EIS"},
    "zphz": {"name": "🔄 Zphz (deg)", "detail": "Sudut fase. Negatif = kapasitif (normal).", "range": "-90-90", "good": "-10°-0°", "bad": "Mendekati 0° atau positif", "source": "Alat EIS"},
    "zreal": {"name": "📉 Zreal (Ohm)", "detail": "Resistansi nyata. Semakin tinggi, semakin rusak.", "range": "0.005-0.02", "good": "Rendah", "bad": "Tinggi", "source": "Alat EIS"},
    "zimg": {"name": "🌀 Zimg (Ohm)", "detail": "Reaktansi imajiner. Negatif = normal.", "range": "-0.01-0.01", "good": "Negatif", "bad": "Mendekati 0 atau positif", "source": "Alat EIS"}
}

# ==================== ANALISIS PER PARAMETER ====================
def analyze_param(param_key, value):
    if param_key == "cycle":
        if value < 300: return "good", "Masih rendah, baterai relatif baru", "Lanjutkan pemakaian normal"
        elif value < 600: return "service", "Mulai memasuki fase menua", "Pantau performa berkala"
        else: return "replace", "Sudah tinggi, mendekati akhir masa pakai", "Segera lakukan inspeksi"
    elif param_key == "soc":
        if 20 <= value <= 80: return "good", "Dalam rentang optimal (20-80%)", "Kebiasaan pengisian baik"
        else: return "service", "Di luar rentang optimal", "Sesuaikan kebiasaan charge"
    elif param_key == "rint":
        if value <= 110: return "good", "Normal, hambatan internal baik", "Pertahankan kondisi"
        elif value <= 150: return "service", "Mulai meningkat, indikasi degradasi", "Lakukan balancing cell"
        else: return "replace", "Sangat tinggi! Baterai rusak", "Segera ganti baterai"
    elif param_key == "ocv":
        if value >= 3.9: return "good", "Normal, tegangan sehat", "Tidak perlu tindakan"
        elif value >= 3.7: return "service", "Mulai menurun", "Periksa sistem pengisian"
        else: return "replace", "Sangat rendah! Sel bermasalah", "Segera ganti baterai"
    elif param_key == "zmod":
        if value <= 0.015: return "good", "Rendah, impedansi normal", "Normal"
        elif value <= 0.025: return "service", "Mulai meningkat", "Monitor berkala"
        else: return "replace", "Tinggi, impedansi membesar", "Indikasi kerusakan"
    elif param_key == "zphz":
        if value <= -5: return "good", "Negatif (kapasitif), normal", "Normal"
        elif value <= 0: return "service", "Mendekati 0°, mulai berubah", "Periksa kondisi"
        else: return "replace", "Positif, sifat berubah", "Indikasi kerusakan"
    elif param_key == "zreal":
        if value <= 0.013: return "good", "Rendah, resistansi normal", "Normal"
        elif value <= 0.018: return "service", "Meningkat, perlu diwaspadai", "Monitor berkala"
        else: return "replace", "Tinggi, hambatan besar", "Segera ganti"
    elif param_key == "zimg":
        if value < -0.002: return "good", "Negatif (kapasitif), normal", "Normal"
        elif value <= 0: return "service", "Mendekati 0, mulai berubah", "Periksa kondisi"
        else: return "replace", "Positif, sifat berubah", "Indikasi kerusakan"
    else:
        return "good", "Parameter normal", "Normal"

# ==================== SESSION STATE ====================
if "step" not in st.session_state:
    st.session_state.step = "cycle"
    st.session_state.data = {}
    st.session_state.messages = []

# ==================== HEADER ====================
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <span style="font-size: 2.5rem;">🔋</span>
    <h1 style="color: white; margin: 0;">Battery Assistant</h1>
    <p style="color: #888;">AI-based SOH Prediction | 9 Parameters</p>
</div>
""", unsafe_allow_html=True)

# ==================== CHAT DISPLAY ====================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "bot":
        # Jika pesan adalah hasil prediksi (sudah dalam bentuk HTML rapi dari fungsi display_result)
        if msg["content"].startswith("###"):
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message-wrapper">
                <div class="bot-avatar">🤖</div>
                <div class="bot-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== FUNGSI INPUT ====================
def show_input(param_key, label, placeholder, step_name):
    info = PARAM_INFO[param_key]
    col1, col2 = st.columns([4, 1])
    with col1:
        val = st.text_input(label, placeholder=placeholder, key=f"input_{param_key}", label_visibility="collapsed")
    with col2:
        with st.popover("ℹ️", use_container_width=True):
            st.markdown(f"**{info['name']}**")
            st.markdown(f"📖 {info['detail']}")
            st.markdown(f"📊 **Rentang:** {info['range']}")
            if info.get('good'):
                st.markdown(f"✅ **Nilai Baik:** {info['good']}")
            if info.get('bad'):
                st.markdown(f"❌ **Nilai Buruk:** {info['bad']}")
            if info.get('note'):
                st.markdown(f"📌 **Catatan:** {info['note']}")
            st.markdown(f"🔌 **Sumber:** {info['source']}")
    return val, step_name

# ==================== STEP 1-9 ====================
current_step = st.session_state.step

if current_step == "cycle":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>Aging Cycle</strong> (0-2000)</div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("cycle", "", "Contoh: 100", "soc")
    if st.button("✅ Kirim", key="send_cycle"):
        try:
            val = float(user_val)
            if 0 <= val <= 2000:
                st.session_state.data["cycle"] = val
                st.session_state.messages.append({"role": "user", "content": f"Aging cycle: {val:.0f}"})
                st.session_state.step = "soc"
                st.rerun()
            else:
                st.error("Masukkan 0-2000")
        except:
            st.error("Masukkan angka valid")

elif current_step == "soc":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>SOC (%)</strong> (0-100)</div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("soc", "", "Contoh: 80", "rint")
    if st.button("✅ Kirim", key="send_soc"):
        try:
            val = float(user_val)
            if 0 <= val <= 100:
                st.session_state.data["soc"] = val
                st.session_state.messages.append({"role": "user", "content": f"SOC: {val:.0f}%"})
                st.session_state.step = "rint"
                st.rerun()
            else:
                st.error("Masukkan 0-100")
        except:
            st.error("Masukkan angka valid")

elif current_step == "rint":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>R_int (%)</strong> (0-200)</div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("rint", "", "Contoh: 100", "ocv")
    if st.button("✅ Kirim", key="send_rint"):
        try:
            val = float(user_val)
            if 0 <= val <= 200:
                st.session_state.data["rint"] = val
                st.session_state.messages.append({"role": "user", "content": f"R_int: {val:.0f}%"})
                st.session_state.step = "ocv"
                st.rerun()
            else:
                st.error("Masukkan 0-200")
        except:
            st.error("Masukkan angka valid")

elif current_step == "ocv":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>OCV (V)</strong> (3.0-4.5)</div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("ocv", "", "Contoh: 4.15", "freq")
    if st.button("✅ Kirim", key="send_ocv"):
        try:
            val = float(user_val)
            if 3.0 <= val <= 4.5:
                st.session_state.data["ocv"] = val
                st.session_state.messages.append({"role": "user", "content": f"OCV: {val:.2f}V"})
                st.session_state.step = "freq"
                st.rerun()
            else:
                st.error("Masukkan 3.0-4.5")
        except:
            st.error("Masukkan angka valid")

elif current_step == "freq":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>Frequency (Hz)</strong> (0.1-10000)</div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("freq", "", "Contoh: 10", "zmod")
    if st.button("✅ Kirim", key="send_freq"):
        try:
            val = float(user_val)
            if 0.1 <= val <= 10000:
                st.session_state.data["freq"] = val
                st.session_state.messages.append({"role": "user", "content": f"Frequency: {val:.2f} Hz"})
                st.session_state.step = "zmod"
                st.rerun()
            else:
                st.error("Masukkan 0.1-10000")
        except:
            st.error("Masukkan angka valid")

elif current_step == "zmod":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>Zmod (Ohm)</strong></div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("zmod", "", "Contoh: 0.012", "zphz")
    if st.button("✅ Kirim", key="send_zmod"):
        try:
            val = float(user_val)
            st.session_state.data["zmod"] = val
            st.session_state.messages.append({"role": "user", "content": f"Zmod: {val:.6f} Ohm"})
            st.session_state.step = "zphz"
            st.rerun()
        except:
            st.error("Masukkan angka valid")

elif current_step == "zphz":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>Zphz (deg)</strong></div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("zphz", "", "Contoh: -2.5", "zreal")
    if st.button("✅ Kirim", key="send_zphz"):
        try:
            val = float(user_val)
            st.session_state.data["zphz"] = val
            st.session_state.messages.append({"role": "user", "content": f"Zphz: {val:.2f}°"})
            st.session_state.step = "zreal"
            st.rerun()
        except:
            st.error("Masukkan angka valid")

elif current_step == "zreal":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>Zreal (Ohm)</strong></div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("zreal", "", "Contoh: 0.012", "zimg")
    if st.button("✅ Kirim", key="send_zreal"):
        try:
            val = float(user_val)
            st.session_state.data["zreal"] = val
            st.session_state.messages.append({"role": "user", "content": f"Zreal: {val:.6f} Ohm"})
            st.session_state.step = "zimg"
            st.rerun()
        except:
            st.error("Masukkan angka valid")

elif current_step == "zimg":
    st.markdown('<div class="bot-message-wrapper"><div class="bot-avatar">🤖</div><div class="bot-bubble">Masukkan <strong>Zimg (Ohm)</strong></div></div>', unsafe_allow_html=True)
    user_val, next_step = show_input("zimg", "", "Contoh: -0.002", "done")
    if st.button("✅ Kirim", key="send_zimg"):
        try:
            val = float(user_val)
            st.session_state.data["zimg"] = val
            st.session_state.messages.append({"role": "user", "content": f"Zimg: {val:.6f} Ohm"})
            
            # ========== PREDIKSI SOH ==========
            input_data = np.array([[
                st.session_state.data["cycle"], st.session_state.data["soc"], st.session_state.data["rint"],
                st.session_state.data["ocv"], st.session_state.data["freq"], st.session_state.data["zmod"],
                st.session_state.data["zphz"], st.session_state.data["zreal"], st.session_state.data["zimg"]
            ]])
            input_scaled = scaler_X.transform(input_data)
            soh = scaler_y.inverse_transform(model.predict(input_scaled).reshape(-1, 1))[0][0]
            
            # Status SOH
            if soh >= 90:
                status = "SEHAT"
                status_color = "green"
            elif soh >= 70:
                status = "WASPADA"
                status_color = "orange"
            else:
                status = "KRITIS"
                status_color = "red"
            
            # Analisis per parameter
            good_count = 0
            service_count = 0
            replace_count = 0
            analysis_items = []
            
            param_order = ["cycle", "soc", "rint", "ocv", "zmod", "zphz", "zreal", "zimg"]
            for p in param_order:
                status_p, msg_p, action_p = analyze_param(p, st.session_state.data[p])
                if status_p == "good":
                    good_count += 1
                    icon = "🟢"
                elif status_p == "service":
                    service_count += 1
                    icon = "🟡"
                else:
                    replace_count += 1
                    icon = "🔴"
                analysis_items.append(f"{icon} **{PARAM_INFO[p]['name']}** = {st.session_state.data[p]} → {msg_p}")
            
            # Buat output RAPI dengan st.container
            result_container = st.container()
            with result_container:
                st.markdown("---")
                st.markdown(f"### 🔋 HASIL PREDIKSI SOH: **{soh:.1f}%**")
                st.markdown(f"#### Status: <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
                
                st.markdown("### 📊 Analisis Per Parameter")
                for item in analysis_items:
                    st.markdown(item)
                
                st.markdown("---")
                st.markdown("### 📋 Ringkasan Rekomendasi")
                st.markdown(f"✅ **LANJUTKAN:** {good_count} parameter")
                st.markdown(f"⚠️ **SERVICE:** {service_count} parameter")
                st.markdown(f"🔴 **GANTI:** {replace_count} parameter")
                
                if replace_count > 0:
                    st.error(f"🔴 **KESIMPULAN:** Ada {replace_count} parameter yang harus GANTI. Segera lakukan tindakan!")
                elif service_count > 0:
                    st.warning(f"⚠️ **KESIMPULAN:** Ada {service_count} parameter yang perlu SERVICE. Lakukan inspeksi segera.")
                else:
                    st.success(f"✅ **KESIMPULAN:** Semua parameter dalam kondisi baik. Lanjutkan pemakaian normal.")
                
                st.caption("📊 Berdasarkan 9 parameter yang dimasukkan")
            
            st.session_state.step = "done"
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🤖 Battery Assistant")
    st.markdown("---")
    st.info("""
    **Metode:** ANN (MLPRegressor)
    **Dataset:** CNR Italy EIS (2026)
    **Output:** SOH + Analisis per Parameter
    """)
    st.markdown("### 📊 Status")
    st.markdown("- 🟢 **≥90%** → SEHAT\n- 🟡 **70-90%** → WASPADA\n- 🔴 **<70%** → KRITIS")
    st.markdown("### 🔧 Tindakan\n- ✅ LANJUTKAN\n- ⚠️ SERVICE\n- 🔴 GANTI")
    if st.button("🔄 Mulai Baru", use_container_width=True):
        st.session_state.step = "cycle"
        st.session_state.data = {}
        st.session_state.messages = []
        st.rerun()
    st.caption("© 2026 | Project SC 2026")

st.markdown('<div class="footer">🔋 Battery Assistant — Prediksi SOH + Analisis Per Parameter</div>', unsafe_allow_html=True)
