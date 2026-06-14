import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Battery Assistant", page_icon="🔋", layout="wide")

# ==================== HIDE SIDEBAR ====================
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }
    .stApp { margin-left: 0; }
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0D0D0D; }
    #MainMenu, footer, header { visibility: hidden; }
    
    .chat-container { max-width: 800px; margin: 0 auto; padding: 1rem; }
    
    .bot-message { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 24px; }
    .bot-avatar { width: 45px; height: 45px; background: linear-gradient(135deg, #10B981, #06B6D4); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink: 0; }
    .bot-bubble { background: #1E1E1E; border-radius: 20px 20px 20px 4px; padding: 16px 20px; color: #E0E0E0; border: 1px solid #2E2E2E; max-width: 80%; }
    
    .user-message { display: flex; justify-content: flex-end; margin-bottom: 24px; }
    .user-bubble { background: linear-gradient(135deg, #10B981, #06B6D4); border-radius: 20px 20px 4px 20px; padding: 12px 20px; color: white; max-width: 70%; font-weight: 500; }
    
    .stButton > button { background: linear-gradient(135deg, #10B981, #06B6D4); border: none; border-radius: 30px; padding: 10px 24px; color: white; font-weight: 600; }
    .stTextInput > div > div > input { background: #1E1E1E; border: 1px solid #2E2E2E; border-radius: 30px; padding: 12px 20px; color: white; }
    
    .result-box { background: linear-gradient(135deg, #1E1E1E, #2E2E2E); border-radius: 20px; padding: 20px; text-align: center; margin: 16px 0; }
    .result-soh { font-size: 3rem; font-weight: 800; margin: 0; }
    .result-status { display: inline-block; padding: 6px 16px; border-radius: 99px; font-size: 0.85rem; font-weight: 600; margin-top: 8px; }
    
    .info-popup { background: #1E1E1E; border-radius: 16px; padding: 16px; margin: 16px 0; border-left: 4px solid #10B981; }
    
    .footer { text-align: center; color: #666; font-size: 0.7rem; padding: 20px; margin-top: 80px; }
    hr { border-color: #2E2E2E; margin: 20px 0; }
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
    MODEL_READY = True
except:
    MODEL_READY = False
    st.error("⚠️ Model tidak ditemukan. Pastikan file .pkl ada.")

# ==================== PARAMETER INFO ====================
PARAM_INFO = {
    "cycle": {
        "name": "🔄 Aging Cycle",
        "range": "0 - 2000",
        "normal": "< 300",
        "warning": "300 - 600",
        "danger": "> 600",
        "desc": "Aging Cycle adalah jumlah siklus charge-discharge baterai. 1 siklus = charge 0% ke 100%. Semakin tinggi cycle, baterai semakin aus.",
        "source": "Dari BMS atau alat diagnostik bengkel",
        "example": "100"
    },
    "soc": {
        "name": "🔋 SOC (%)",
        "range": "0 - 100",
        "normal": "20% - 80%",
        "warning": "< 20% atau > 80%",
        "danger": "-",
        "desc": "SOC (State of Charge) adalah level pengisian baterai saat ini. Baterai paling awet di rentang 20-80%.",
        "source": "Dari dashboard mobil listrik",
        "example": "80"
    },
    "rint": {
        "name": "⚡ R_int (%)",
        "range": "0 - 200",
        "normal": "≤ 110%",
        "warning": "110% - 150%",
        "danger": "> 150%",
        "desc": "R_int (Internal Resistance) adalah hambatan listrik di dalam baterai. Baterai sehat punya R_int sekitar 100%. Semakin tinggi, semakin rusak.",
        "source": "Dari alat diagnostik / BMS",
        "example": "100"
    },
    "ocv": {
        "name": "🔌 OCV (V)",
        "range": "3.0 - 4.5",
        "normal": "≥ 3.9V",
        "warning": "3.7V - 3.9V",
        "danger": "< 3.7V",
        "desc": "OCV (Open Circuit Voltage) adalah tegangan baterai saat tidak dipakai. Baterai sehat memiliki OCV 3.8V - 4.2V.",
        "source": "Dari voltmeter / BMS",
        "example": "4.15"
    },
    "freq": {
        "name": "📊 Frequency (Hz)",
        "range": "0.1 - 10000",
        "normal": "-",
        "warning": "-",
        "danger": "-",
        "desc": "Frequency adalah frekuensi pengukuran EIS. Parameter teknis, bukan indikator kesehatan baterai.",
        "source": "Dari alat EIS",
        "example": "10",
        "note": "Parameter teknis"
    },
    "zmod": {
        "name": "📈 Zmod (Ohm)",
        "range": "0.005 - 0.05",
        "normal": "≤ 0.015",
        "warning": "0.015 - 0.025",
        "danger": "> 0.025",
        "desc": "Zmod adalah modulus impedansi (besaran impedansi total baterai). Semakin tinggi, semakin besar degradasi.",
        "source": "Dari alat EIS",
        "example": "0.012"
    },
    "zphz": {
        "name": "🔄 Zphz (deg)",
        "range": "-90 - 90",
        "normal": "≤ -5°",
        "warning": "-5° - 0°",
        "danger": "> 0°",
        "desc": "Zphz adalah sudut fase impedansi. Baterai sehat memiliki sudut negatif (kapasitif).",
        "source": "Dari alat EIS",
        "example": "-2.5"
    },
    "zreal": {
        "name": "📉 Zreal (Ohm)",
        "range": "0.005 - 0.02",
        "normal": "≤ 0.013",
        "warning": "0.013 - 0.018",
        "danger": "> 0.018",
        "desc": "Zreal adalah komponen resistif dari impedansi. Berkorelasi dengan R_int. Semakin tinggi, semakin besar hambatan.",
        "source": "Dari alat EIS",
        "example": "0.012"
    },
    "zimg": {
        "name": "🌀 Zimg (Ohm)",
        "range": "-0.01 - 0.01",
        "normal": "< -0.002",
        "warning": "-0.002 - 0",
        "danger": "> 0",
        "desc": "Zimg adalah komponen reaktif dari impedansi. Nilai negatif = kapasitif (normal).",
        "source": "Dari alat EIS",
        "example": "-0.002"
    }
}

# ==================== ANALISIS ====================
def analyze(key, val):
    if key == "cycle":
        if val < 300: return "good", f"{val} (Masih baru)", "Lanjutkan normal"
        elif val < 600: return "warning", f"{val} (Mulai menua)", "Pantau berkala"
        else: return "danger", f"{val} (Sudah tinggi)", "Inspeksi segera"
    elif key == "soc":
        if 20 <= val <= 80: return "good", f"{val}% (Optimal)", "Kebiasaan baik"
        else: return "warning", f"{val}% (Tidak ideal)", "Sesuaikan charge"
    elif key == "rint":
        if val <= 110: return "good", f"{val}% (Normal)", "Pertahankan"
        elif val <= 150: return "warning", f"{val}% (Mulai naik)", "Balancing cell"
        else: return "danger", f"{val}% (Sangat tinggi)", "Ganti baterai"
    elif key == "ocv":
        if val >= 3.9: return "good", f"{val}V (Normal)", "Tidak perlu tindakan"
        elif val >= 3.7: return "warning", f"{val}V (Mulai turun)", "Periksa sistem"
        else: return "danger", f"{val}V (Sangat rendah)", "Ganti baterai"
    elif key in ["zmod", "zreal"]:
        if val <= 0.015: return "good", f"{val} (Normal)", "Lanjutkan"
        elif val <= 0.025: return "warning", f"{val} (Mulai naik)", "Monitor"
        else: return "danger", f"{val} (Tinggi)", "Indikasi rusak"
    elif key == "zphz":
        if val <= -5: return "good", f"{val}° (Normal)", "Lanjutkan"
        elif val <= 0: return "warning", f"{val}° (Mulai berubah)", "Periksa"
        else: return "danger", f"{val}° (Positif)", "Indikasi rusak"
    elif key == "zimg":
        if val < -0.002: return "good", f"{val} (Normal)", "Lanjutkan"
        elif val <= 0: return "warning", f"{val} (Mulai berubah)", "Periksa"
        else: return "danger", f"{val} (Positif)", "Indikasi rusak"
    return "good", str(val), "Normal"

# ==================== SESSION STATE ====================
if "step" not in st.session_state:
    st.session_state.step = "cycle"
    st.session_state.data = {}
    st.session_state.messages = [{"role": "bot", "content": "👋 Halo! Aku **Battery Assistant** 🤖\n\nAyo cek kesehatan baterai mobil listrikmu!\n\nMasukkan **Aging Cycle** (0-2000)\n💡 *Contoh: 100*"}]
    st.session_state.show_info = None
    st.session_state.result = None
    st.session_state.input_key = 0

# ==================== HEADER ====================
st.markdown('<div style="text-align: center; padding: 20px 0 10px 0;"><span style="font-size: 2.5rem;">🔋</span><h1 style="color: white; margin: 0;">Battery Assistant</h1><p style="color: #888;">AI-based SOH Prediction | ANN Model</p></div>', unsafe_allow_html=True)

# ==================== CHAT DISPLAY ====================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "bot":
        st.markdown(f'<div class="bot-message"><div class="bot-avatar">🤖</div><div class="bot-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="user-message"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

# ==================== POPUP INFO ====================
if st.session_state.show_info and st.session_state.show_info in PARAM_INFO:
    info = PARAM_INFO[st.session_state.show_info]
    
    popup_content = f"""
    <div class="info-popup">
        <strong>ℹ️ {info['name']}</strong><br><br>
        {info['desc']}<br><br>
        📊 <strong>Rentang:</strong> {info['range']}<br>
        ✅ <strong>Normal:</strong> {info['normal']}<br>
        ⚠️ <strong>Waspada:</strong> {info['warning']}<br>
        🔴 <strong>Kritis:</strong> {info['danger']}<br>
        🔌 <strong>Sumber:</strong> {info['source']}<br>
        💡 <strong>Contoh:</strong> {info['example']}
    </div>
    """
    st.markdown(popup_content, unsafe_allow_html=True)
    if st.button("Tutup", key="close_info", use_container_width=True):
        st.session_state.show_info = None
        st.rerun()

# ==================== HASIL PREDIKSI ====================
if st.session_state.result:
    soh, status, good, service, replace, analysis = st.session_state.result
    
    color = "green" if status == "SEHAT" else ("orange" if status == "WASPADA" else "red")
    
    st.markdown(f"""
    <div class="result-box">
        <div class="result-soh" style="color: {color};">{soh:.1f}%</div>
        <div class="result-status" style="background: rgba({('16,185,129' if status=="SEHAT" else ('245,158,11' if status=="WASPADA" else '239,68,68'))}, 0.2); color: {color};">{status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**📊 Analisis Per Parameter:**")
    for name, msg, action, stat in analysis:
        if stat == "good":
            st.success(f"✅ **{name}** : {msg} → {action}")
        elif stat == "warning":
            st.warning(f"⚠️ **{name}** : {msg} → {action}")
        else:
            st.error(f"🔴 **{name}** : {msg} → {action}")
    
    st.markdown("---")
    st.markdown("**📋 RINGKASAN:**")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("✅ LANJUTKAN", f"{good} parameter")
    with c2: st.metric("⚠️ SERVICE", f"{service} parameter")
    with c3: st.metric("🔴 GANTI", f"{replace} parameter")
    
    if replace > 0:
        st.error(f"🔴 **KESIMPULAN:** Ada {replace} parameter yang harus GANTI. Segera lakukan tindakan!")
    elif service > 0:
        st.warning(f"⚠️ **KESIMPULAN:** Ada {service} parameter yang perlu SERVICE. Lakukan inspeksi segera.")
    else:
        st.success(f"✅ **KESIMPULAN:** Semua parameter dalam kondisi baik. Lanjutkan pemakaian normal.")
    
    st.caption("📊 Berdasarkan 9 parameter yang dimasukkan")
    
    if st.button("🔄 Mulai Baru", use_container_width=True):
        st.session_state.step = "cycle"
        st.session_state.data = {}
        st.session_state.messages = [{"role": "bot", "content": "👋 Halo! Aku **Battery Assistant** 🤖\n\nAyo cek kesehatan baterai mobil listrikmu!\n\nMasukkan **Aging Cycle** (0-2000)\n💡 *Contoh: 100*"}]
        st.session_state.result = None
        st.session_state.show_info = None
        st.session_state.input_key = 0
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ==================== INPUT FORM ====================
if st.session_state.result is None:
    step = st.session_state.step
    info = PARAM_INFO[step]
    
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        st.markdown(f'<p style="color: #888; font-size: 0.75rem; margin-bottom: 4px;">💡 Contoh: {info["example"]}</p>', unsafe_allow_html=True)
        # Key dinamis agar input kosong setiap kali berganti step
        user_input = st.text_input("", key=f"input_{st.session_state.input_key}", placeholder=f"Ketik {info['name']}...", label_visibility="collapsed")
    with col2:
        if st.button("ℹ️", key="info_btn", use_container_width=True):
            st.session_state.show_info = step
    with col3:
        submitted = st.button("✅ Kirim", key="send_btn", use_container_width=True)
    
    if submitted and user_input:
        st.session_state.show_info = None
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            val = float(user_input)
            
            if step == "cycle":
                if 0 <= val <= 2000:
                    st.session_state.data["cycle"] = val
                    st.session_state.step = "soc"
                    st.session_state.input_key += 1  # Reset input box
                    st.session_state.messages.append({"role": "bot", "content": f"✅ Aging cycle: {val:.0f}\n\nMasukkan **SOC (%)** (0-100)\n💡 *Contoh: 80*"})
                else:
                    st.session_state.messages.append({"role": "bot", "content": f"⚠️ Masukkan angka 0-2000. Contoh: 100"})
            
            elif step == "soc":
                if 0 <= val <= 100:
                    st.session_state.data["soc"] = val
                    st.session_state.step = "rint"
                    st.session_state.input_key += 1
                    st.session_state.messages.append({"role": "bot", "content": f"✅ SOC: {val:.0f}%\n\nMasukkan **R_int (%)** (0-200)\n💡 *Contoh: 100*"})
                else:
                    st.session_state.messages.append({"role": "bot", "content": f"⚠️ Masukkan angka 0-100. Contoh: 80"})
            
            elif step == "rint":
                if 0 <= val <= 200:
                    st.session_state.data["rint"] = val
                    st.session_state.step = "ocv"
                    st.session_state.input_key += 1
                    st.session_state.messages.append({"role": "bot", "content": f"✅ R_int: {val:.0f}%\n\nMasukkan **OCV (V)** (3.0-4.5)\n💡 *Contoh: 4.15*"})
                else:
                    st.session_state.messages.append({"role": "bot", "content": f"⚠️ Masukkan angka 0-200. Contoh: 100"})
            
            elif step == "ocv":
                if 3.0 <= val <= 4.5:
                    st.session_state.data["ocv"] = val
                    st.session_state.step = "freq"
                    st.session_state.input_key += 1
                    st.session_state.messages.append({"role": "bot", "content": f"✅ OCV: {val:.2f}V\n\nMasukkan **Frequency (Hz)** (0.1-10000)\n💡 *Contoh: 10*"})
                else:
                    st.session_state.messages.append({"role": "bot", "content": f"⚠️ Masukkan angka 3.0-4.5. Contoh: 4.15"})
            
            elif step == "freq":
                if 0.1 <= val <= 10000:
                    st.session_state.data["freq"] = val
                    st.session_state.step = "zmod"
                    st.session_state.input_key += 1
                    st.session_state.messages.append({"role": "bot", "content": f"✅ Frequency: {val:.2f} Hz\n\nMasukkan **Zmod (Ohm)** (0.005-0.05)\n💡 *Contoh: 0.012*"})
                else:
                    st.session_state.messages.append({"role": "bot", "content": f"⚠️ Masukkan angka 0.1-10000. Contoh: 10"})
            
            elif step == "zmod":
                st.session_state.data["zmod"] = val
                st.session_state.step = "zphz"
                st.session_state.input_key += 1
                st.session_state.messages.append({"role": "bot", "content": f"✅ Zmod: {val:.6f} Ohm\n\nMasukkan **Zphz (deg)** (-90-90)\n💡 *Contoh: -2.5*"})
            
            elif step == "zphz":
                st.session_state.data["zphz"] = val
                st.session_state.step = "zreal"
                st.session_state.input_key += 1
                st.session_state.messages.append({"role": "bot", "content": f"✅ Zphz: {val:.2f}°\n\nMasukkan **Zreal (Ohm)** (0.005-0.02)\n💡 *Contoh: 0.012*"})
            
            elif step == "zreal":
                st.session_state.data["zreal"] = val
                st.session_state.step = "zimg"
                st.session_state.input_key += 1
                st.session_state.messages.append({"role": "bot", "content": f"✅ Zreal: {val:.6f} Ohm\n\nMasukkan **Zimg (Ohm)** (-0.01-0.01)\n💡 *Contoh: -0.002*"})
            
            elif step == "zimg":
                st.session_state.data["zimg"] = val
                
                if MODEL_READY:
                    input_data = np.array([[
                        st.session_state.data["cycle"], st.session_state.data["soc"],
                        st.session_state.data["rint"], st.session_state.data["ocv"],
                        st.session_state.data["freq"], st.session_state.data["zmod"],
                        st.session_state.data["zphz"], st.session_state.data["zreal"],
                        st.session_state.data["zimg"]
                    ]])
                    input_scaled = scaler_X.transform(input_data)
                    soh = scaler_y.inverse_transform(model.predict(input_scaled).reshape(-1, 1))[0][0]
                    
                    if soh >= 90: status = "SEHAT"
                    elif soh >= 70: status = "WASPADA"
                    else: status = "KRITIS"
                    
                    params = ["cycle", "soc", "rint", "ocv", "zmod", "zphz", "zreal", "zimg"]
                    good = service = replace = 0
                    analysis = []
                    
                    for p in params:
                        stat, msg, action = analyze(p, st.session_state.data[p])
                        if stat == "good": good += 1
                        elif stat == "warning": service += 1
                        else: replace += 1
                        analysis.append((PARAM_INFO[p]['name'], msg, action, stat))
                    
                    st.session_state.result = (soh, status, good, service, replace, analysis)
                    st.session_state.messages.append({"role": "bot", "content": "✅ Semua data sudah dimasukkan! Lihat hasil analisis di bawah."})
                else:
                    st.session_state.messages.append({"role": "bot", "content": "⚠️ Model tidak tersedia"})
        
        except ValueError:
            st.session_state.messages.append({"role": "bot", "content": f"⚠️ Masukkan angka yang valid. Contoh: {info['example']}"})
        
        st.rerun()

# ==================== FOOTER ====================
st.markdown('<div class="footer">🔋 Battery Assistant — Prediksi SOH + Analisis Per Parameter © 2026</div>', unsafe_allow_html=True)
