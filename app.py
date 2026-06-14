import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Battery Assistant", page_icon="🔋", layout="wide")

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0D0D0D; }
    #MainMenu, footer, header { visibility: hidden; }
    .chat-container { max-width: 900px; margin: 0 auto; padding: 1rem; }
    .bot-message-wrapper { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 24px; animation: fadeIn 0.3s ease; }
    .bot-avatar { width: 45px; height: 45px; background: linear-gradient(135deg, #10B981, #06B6D4); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink: 0; }
    .bot-bubble { background: #1E1E1E; border-radius: 20px 20px 20px 4px; padding: 16px 20px; color: #E0E0E0; border: 1px solid #2E2E2E; max-width: 80%; }
    .user-message { display: flex; justify-content: flex-end; margin-bottom: 24px; animation: fadeIn 0.3s ease; }
    .user-bubble { background: linear-gradient(135deg, #10B981, #06B6D4); border-radius: 20px 20px 4px 20px; padding: 12px 20px; color: white; max-width: 70%; font-weight: 500; }
    .result-card { background: linear-gradient(135deg, #1E1E1E, #2E2E2E); border-radius: 20px; padding: 20px; text-align: center; margin: 16px 0; }
    .soh-value { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #10B981, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .status-badge { display: inline-block; padding: 6px 16px; border-radius: 99px; font-size: 0.85rem; font-weight: 600; margin-top: 8px; }
    .healthy { background: rgba(16, 185, 129, 0.2); color: #10B981; }
    .warning { background: rgba(245, 158, 11, 0.2); color: #F59E0B; }
    .critical { background: rgba(239, 68, 68, 0.2); color: #EF4444; }
    .analysis-card { border-radius: 12px; padding: 12px; margin: 8px 0; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .footer { text-align: center; color: #666; font-size: 0.7rem; padding: 20px; margin-top: 100px; }
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
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model tidak ditemukan")

# ==================== PARAMETER INFO ====================
PARAM_INFO = {
    "cycle": {"name": "🔄 Aging Cycle", "desc": "Jumlah siklus charge-discharge", "range": "0-2000", "good": "<300", "bad": ">600", "detail": "1 siklus = charge 0% ke 100%"},
    "soc": {"name": "🔋 SOC (%)", "desc": "Level pengisian baterai", "range": "0-100", "good": "20-80%", "bad": "<20% atau >80%", "detail": "Ideal 20-80%"},
    "rint": {"name": "⚡ R_int (%)", "desc": "Internal resistance", "range": "0-200", "good": "≤110%", "bad": ">150%", "detail": "Semakin tinggi semakin rusak"},
    "ocv": {"name": "🔌 OCV (V)", "desc": "Tegangan diam baterai", "range": "3.0-4.5", "good": "≥3.9V", "bad": "<3.7V", "detail": "Sehat di atas 3.8V"},
    "freq": {"name": "📊 Frequency (Hz)", "desc": "Frekuensi pengukuran EIS", "range": "0.1-10000", "good": None, "bad": None, "detail": "Parameter teknis", "note": "Parameter pengukuran"},
    "zmod": {"name": "📈 Zmod (Ohm)", "desc": "Modulus impedansi", "range": "0.005-0.05", "good": "≤0.015", "bad": ">0.025", "detail": "Meningkat seiring degradasi"},
    "zphz": {"name": "🔄 Zphz (deg)", "desc": "Sudut fase impedansi", "range": "-90-90", "good": "≤-5°", "bad": ">0°", "detail": "Negatif = normal"},
    "zreal": {"name": "📉 Zreal (Ohm)", "desc": "Resistansi nyata", "range": "0.005-0.02", "good": "≤0.013", "bad": ">0.018", "detail": "Berkorelasi dengan R_int"},
    "zimg": {"name": "🌀 Zimg (Ohm)", "desc": "Reaktansi imajiner", "range": "-0.01-0.01", "good": "< -0.002", "bad": ">0", "detail": "Negatif = normal"}
}

# ==================== FUNGSI ANALISIS ====================
def analyze_parameter(param_key, value):
    if param_key == "cycle":
        if value < 300: return "good", f"{value} (Masih rendah, baterai baru)", "Lanjutkan normal"
        elif value < 600: return "service", f"{value} (Mulai menua)", "Pantau berkala"
        else: return "replace", f"{value} (Sudah tinggi)", "Inspeksi segera"
    elif param_key == "soc":
        if 20 <= value <= 80: return "good", f"{value}% (Optimal)", "Kebiasaan baik"
        else: return "service", f"{value}% (Tidak ideal)", "Sesuaikan charge"
    elif param_key == "rint":
        if value <= 110: return "good", f"{value}% (Normal)", "Pertahankan"
        elif value <= 150: return "service", f"{value}% (Mulai naik)", "Balancing cell"
        else: return "replace", f"{value}% (Sangat tinggi)", "Ganti baterai"
    elif param_key == "ocv":
        if value >= 3.9: return "good", f"{value}V (Normal)", "Tidak perlu tindakan"
        elif value >= 3.7: return "service", f"{value}V (Mulai turun)", "Periksa sistem"
        else: return "replace", f"{value}V (Sangat rendah)", "Ganti baterai"
    elif param_key == "zmod":
        if value <= 0.015: return "good", f"{value} Ohm (Normal)", "Lanjutkan"
        elif value <= 0.025: return "service", f"{value} Ohm (Mulai naik)", "Monitor"
        else: return "replace", f"{value} Ohm (Tinggi)", "Indikasi rusak"
    elif param_key == "zphz":
        if value <= -5: return "good", f"{value}° (Normal)", "Lanjutkan"
        elif value <= 0: return "service", f"{value}° (Mulai berubah)", "Periksa"
        else: return "replace", f"{value}° (Positif)", "Indikasi rusak"
    elif param_key == "zreal":
        if value <= 0.013: return "good", f"{value} Ohm (Normal)", "Lanjutkan"
        elif value <= 0.018: return "service", f"{value} Ohm (Mulai naik)", "Monitor"
        else: return "replace", f"{value} Ohm (Tinggi)", "Indikasi rusak"
    elif param_key == "zimg":
        if value < -0.002: return "good", f"{value} Ohm (Normal)", "Lanjutkan"
        elif value <= 0: return "service", f"{value} Ohm (Mulai berubah)", "Periksa"
        else: return "replace", f"{value} Ohm (Positif)", "Indikasi rusak"
    return "good", str(value), "Normal"

# ==================== SESSION STATE ====================
if "step" not in st.session_state:
    st.session_state.step = "cycle"
    st.session_state.data = {}
    st.session_state.messages = []

# ==================== HEADER ====================
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <span style="font-size: 2.5rem;">🔋</span>
    <h1 style="color: white; margin: 0; font-size: 1.8rem;">Battery Assistant</h1>
    <p style="color: #888; font-size: 0.8rem;">AI-based State of Health (SOH) Prediction | 9 Parameters | ANN Model</p>
</div>
""", unsafe_allow_html=True)

# ==================== CHAT DISPLAY ====================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "bot":
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
def show_input(param_key, placeholder, input_key):
    info = PARAM_INFO[param_key]
    col1, col2 = st.columns([4, 1])
    with col1:
        val = st.text_input("", placeholder=placeholder, key=input_key, label_visibility="collapsed")
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
    return val

# ==================== STEP 1-9 ====================
current_step = st.session_state.step

if current_step == "cycle":
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>Aging Cycle</strong> (0-2000)<br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['cycle']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("cycle", "Contoh: 100", "cycle_input")
    if st.button("✅ Kirim", key="send_cycle"):
        if user_val:
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
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>SOC (%)</strong> (0-100)<br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['soc']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("soc", "Contoh: 80", "soc_input")
    if st.button("✅ Kirim", key="send_soc"):
        if user_val:
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
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>R_int (%)</strong> (0-200)<br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['rint']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("rint", "Contoh: 100", "rint_input")
    if st.button("✅ Kirim", key="send_rint"):
        if user_val:
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
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>OCV (V)</strong> (3.0-4.5)<br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['ocv']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("ocv", "Contoh: 4.15", "ocv_input")
    if st.button("✅ Kirim", key="send_ocv"):
        if user_val:
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
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>Frequency (Hz)</strong> (0.1-10000)<br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['freq']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("freq", "Contoh: 10", "freq_input")
    if st.button("✅ Kirim", key="send_freq"):
        if user_val:
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
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>Zmod (Ohm)</strong><br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['zmod']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("zmod", "Contoh: 0.012", "zmod_input")
    if st.button("✅ Kirim", key="send_zmod"):
        if user_val:
            try:
                val = float(user_val)
                st.session_state.data["zmod"] = val
                st.session_state.messages.append({"role": "user", "content": f"Zmod: {val:.6f} Ohm"})
                st.session_state.step = "zphz"
                st.rerun()
            except:
                st.error("Masukkan angka valid")

elif current_step == "zphz":
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>Zphz (deg)</strong><br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['zphz']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("zphz", "Contoh: -2.5", "zphz_input")
    if st.button("✅ Kirim", key="send_zphz"):
        if user_val:
            try:
                val = float(user_val)
                st.session_state.data["zphz"] = val
                st.session_state.messages.append({"role": "user", "content": f"Zphz: {val:.2f}°"})
                st.session_state.step = "zreal"
                st.rerun()
            except:
                st.error("Masukkan angka valid")

elif current_step == "zreal":
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>Zreal (Ohm)</strong><br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['zreal']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("zreal", "Contoh: 0.012", "zreal_input")
    if st.button("✅ Kirim", key="send_zreal"):
        if user_val:
            try:
                val = float(user_val)
                st.session_state.data["zreal"] = val
                st.session_state.messages.append({"role": "user", "content": f"Zreal: {val:.6f} Ohm"})
                st.session_state.step = "zimg"
                st.rerun()
            except:
                st.error("Masukkan angka valid")

elif current_step == "zimg":
    st.markdown(f"""
    <div class="bot-message-wrapper">
        <div class="bot-avatar">🤖</div>
        <div class="bot-bubble">
            Masukkan <strong>Zimg (Ohm)</strong><br>
            <span style="font-size: 0.7rem; color: #888;">{PARAM_INFO['zimg']['desc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    user_val = show_input("zimg", "Contoh: -0.002", "zimg_input")
    if st.button("✅ Kirim", key="send_zimg"):
        if user_val:
            try:
                val = float(user_val)
                st.session_state.data["zimg"] = val
                st.session_state.messages.append({"role": "user", "content": f"Zimg: {val:.6f} Ohm"})
                
                if model_loaded:
                    input_data = np.array([[
                        st.session_state.data["cycle"],
                        st.session_state.data["soc"],
                        st.session_state.data["rint"],
                        st.session_state.data["ocv"],
                        st.session_state.data["freq"],
                        st.session_state.data["zmod"],
                        st.session_state.data["zphz"],
                        st.session_state.data["zreal"],
                        st.session_state.data["zimg"]
                    ]])
                    input_scaled = scaler_X.transform(input_data)
                    soh = scaler_y.inverse_transform(model.predict(input_scaled).reshape(-1, 1))[0][0]
                    
                    if soh >= 90:
                        status = "SEHAT"
                        status_class = "healthy"
                    elif soh >= 70:
                        status = "WASPADA"
                        status_class = "warning"
                    else:
                        status = "KRITIS"
                        status_class = "critical"
                    
                    params = ["cycle", "soc", "rint", "ocv", "zmod", "zphz", "zreal", "zimg"]
                    param_status = {"good": 0, "service": 0, "replace": 0}
                    
                    # Kumpulkan hasil analisis
                    result_lines = []
                    result_lines.append(f"""<div class="result-card"><div class="soh-value">{soh:.1f}%</div><div class="status-badge {status_class}">{status}</div><div style="margin-top: 16px;"><strong>📊 Analisis Per Parameter:</strong></div>""")
                    
                    for p in params:
                        stat, msg, action = analyze_parameter(p, st.session_state.data[p])
                        param_status[stat] += 1
                        
                        if stat == "good":
                            bg = "rgba(16, 185, 129, 0.1)"
                            border = "#10B981"
                            label = "✅ LANJUTKAN"
                        elif stat == "service":
                            bg = "rgba(245, 158, 11, 0.1)"
                            border = "#F59E0B"
                            label = "⚠️ SERVICE"
                        else:
                            bg = "rgba(239, 68, 68, 0.1)"
                            border = "#EF4444"
                            label = "🔴 GANTI"
                        
                        result_lines.append(f"""
                        <div style="background: {bg}; border-left: 4px solid {border}; border-radius: 12px; padding: 12px; margin: 8px 0; text-align: left;">
                            <strong>{PARAM_INFO[p]['name']}</strong> = {st.session_state.data[p]}<br>
                            <span>{msg}</span><br>
                            <span style="color: {border}; font-weight: 600;">{label}: {action}</span>
                        </div>
                        """)
                    
                    result_lines.append(f"""
                        <div style="margin-top: 20px; padding: 16px; background: #0D0D0D; border-radius: 16px; text-align: left;">
                            <strong>📋 RINGKASAN:</strong><br>
                            <span style="color: #10B981;">✅ LANJUTKAN: {param_status['good']} parameter</span><br>
                            <span style="color: #F59E0B;">⚠️ SERVICE: {param_status['service']} parameter</span><br>
                            <span style="color: #EF4444;">🔴 GANTI: {param_status['replace']} parameter</span>
                    """)
                    
                    if param_status["replace"] > 0:
                        result_lines.append(f'<br><span style="color: #EF4444;">🔴 KESIMPULAN: Ada {param_status["replace"]} parameter yang harus GANTI.</span>')
                    elif param_status["service"] > 0:
                        result_lines.append(f'<br><span style="color: #F59E0B;">⚠️ KESIMPULAN: Ada {param_status["service"]} parameter yang perlu SERVICE.</span>')
                    else:
                        result_lines.append(f'<br><span style="color: #10B981;">✅ KESIMPULAN: Semua parameter dalam kondisi baik.</span>')
                    
                    result_lines.append("</div></div>")
                    
                    result_html = "".join(result_lines)
                    
                    st.session_state.messages.append({"role": "bot", "content": result_html})
                    st.session_state.step = "done"
                    st.rerun()
                else:
                    st.error("Model tidak tersedia")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🤖 Battery Assistant")
    st.markdown("---")
    st.markdown("### 📋 Panduan Parameter")
    st.markdown("""
    | Parameter | Normal | Service | Ganti |
    |:---|:---|:---|:---|
    | Aging Cycle | <300 | 300-600 | >600 |
    | SOC (%) | 20-80 | di luar | - |
    | R_int (%) | ≤110 | 110-150 | >150 |
    | OCV (V) | ≥3.9 | 3.7-3.9 | <3.7 |
    | Zmod | ≤0.015 | 0.015-0.025 | >0.025 |
    | Zphz | ≤-5 | -5-0 | >0 |
    | Zreal | ≤0.013 | 0.013-0.018 | >0.018 |
    | Zimg | <-0.002 | -0.002-0 | >0 |
    """)
    st.markdown("---")
    if st.button("🔄 Mulai Baru", use_container_width=True):
        st.session_state.step = "cycle"
        st.session_state.data = {}
        st.session_state.messages = []
        st.rerun()
    st.caption("© 2026 | Project SC 2026")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    🔋 Battery Assistant — Prediksi SOH + Analisis Per Parameter + Rekomendasi
</div>
""", unsafe_allow_html=True)
