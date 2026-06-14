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
    
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .bot-message-wrapper {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 24px;
        animation: fadeIn 0.3s ease;
    }
    
    .bot-avatar {
        width: 45px;
        height: 45px;
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
    }
    
    .bot-content {
        display: flex;
        flex-direction: column;
        gap: 8px;
        max-width: 80%;
    }
    
    .bot-bubble {
        background: #1E1E1E;
        border-radius: 20px 20px 20px 4px;
        padding: 16px 20px;
        color: #E0E0E0;
        font-size: 0.95rem;
        line-height: 1.5;
        border: 1px solid #2E2E2E;
    }
    
    .info-button {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        border-radius: 30px;
        padding: 6px 16px;
        color: #10B981;
        font-size: 0.75rem;
        font-weight: 600;
        cursor: pointer;
        width: fit-content;
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 24px;
        animation: fadeIn 0.3s ease;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border-radius: 20px 20px 4px 20px;
        padding: 12px 20px;
        color: white;
        font-size: 0.95rem;
        max-width: 70%;
        font-weight: 500;
    }
    
    .input-wrapper {
        display: flex;
        gap: 10px;
        align-items: center;
        margin-top: 10px;
    }
    
    .input-wrapper input {
        flex: 1;
        background: #1E1E1E;
        border: 1px solid #2E2E2E;
        border-radius: 30px;
        padding: 12px 20px;
        color: white;
        font-size: 0.9rem;
    }
    
    .input-wrapper button {
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border: none;
        border-radius: 30px;
        padding: 12px 24px;
        color: white;
        font-weight: 600;
        cursor: pointer;
    }
    
    .result-card {
        background: linear-gradient(135deg, #1E1E1E, #2E2E2E);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        margin: 16px 0;
    }
    
    .soh-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10B981, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 99px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .healthy { background: rgba(16, 185, 129, 0.2); color: #10B981; }
    .warning { background: rgba(245, 158, 11, 0.2); color: #F59E0B; }
    .critical { background: rgba(239, 68, 68, 0.2); color: #EF4444; }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.7rem;
        padding: 20px;
        margin-top: 100px;
    }
    
    .stPopover {
        background: #1E1E1E !important;
        border: 1px solid #10B981 !important;
        border-radius: 16px !important;
    }
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
    st.error("⚠️ Model tidak ditemukan. Pastikan file model_soh.pkl, scaler_X.pkl, scaler_y.pkl ada.")
    st.stop()

# ==================== PARAMETER INFO ====================
PARAM_INFO = {
    "cycle": {
        "name": "🔄 Aging Cycle",
        "desc": "Jumlah siklus charge-discharge yang sudah dilalui baterai.",
        "detail": "1 siklus = charge dari 0% ke 100% (atau akumulasi). Semakin tinggi cycle, baterai semakin aus.",
        "range": "0 - 2000",
        "good": "Rendah (<300)",
        "bad": "Tinggi (>600)",
        "source": "Dari BMS (Battery Management System) atau alat diagnostik"
    },
    "soc": {
        "name": "🔋 SOC (%)",
        "desc": "State of Charge — level pengisian baterai saat ini.",
        "detail": "SOC ideal untuk kesehatan baterai adalah 20-80%. Terlalu rendah (<20%) atau terlalu tinggi (>80%) mempercepat degradasi.",
        "range": "0 - 100",
        "good": "20-80%",
        "bad": "<20% atau >80%",
        "source": "Dari dashboard mobil listrik"
    },
    "rint": {
        "name": "⚡ R_int (%)",
        "desc": "Internal Resistance — hambatan listrik di dalam baterai.",
        "detail": "Baterai sehat punya R_int sekitar 100%. Semakin tinggi R_int, semakin rusak baterai.",
        "range": "0 - 200",
        "good": "≤110%",
        "bad": ">150%",
        "source": "Dari alat diagnostik / BMS"
    },
    "ocv": {
        "name": "🔌 OCV (V)",
        "desc": "Open Circuit Voltage — tegangan baterai saat tidak dipakai.",
        "detail": "Baterai lithium-ion sehat memiliki OCV antara 3.8V - 4.2V. Di bawah 3.7V mengindikasikan sel rusak.",
        "range": "3.0 - 4.5",
        "good": "≥3.9V",
        "bad": "<3.7V",
        "source": "Dari voltmeter / BMS"
    },
    "freq": {
        "name": "📊 Frequency (Hz)",
        "desc": "Frekuensi pengukuran EIS (Electrochemical Impedance Spectroscopy).",
        "detail": "Parameter teknis pengukuran. Frekuensi rendah (0.1Hz) sensitif ke degradasi, frekuensi tinggi (1000Hz) mengukur resistansi murni.",
        "range": "0.1 - 10000",
        "good": None,
        "bad": None,
        "source": "Dari alat EIS",
        "note": "Parameter pengukuran, bukan indikator kesehatan baterai"
    },
    "zmod": {
        "name": "📈 Zmod (Ohm)",
        "desc": "Modulus impedansi — besaran impedansi total baterai.",
        "detail": "Nilai impedansi total baterai. Meningkat seiring degradasi baterai.",
        "range": "0.005 - 0.05",
        "good": "Rendah (mendekati nilai awal)",
        "bad": "Tinggi (naik >20% dari awal)",
        "source": "Dari alat EIS"
    },
    "zphz": {
        "name": "🔄 Zphz (deg)",
        "desc": "Sudut fase impedansi.",
        "detail": "Menunjukkan sifat baterai. Baterai sehat biasanya memiliki sudut fase negatif (kapasitif).",
        "range": "-90 - 90",
        "good": "-10° - 0°",
        "bad": "Mendekati 0° atau positif",
        "source": "Dari alat EIS"
    },
    "zreal": {
        "name": "📉 Zreal (Ohm)",
        "desc": "Komponen resistif (nyata) dari impedansi.",
        "detail": "Berkorelasi langsung dengan R_int. Semakin tinggi, semakin besar hambatan internal.",
        "range": "0.005 - 0.02",
        "good": "Rendah (mendekati nilai awal)",
        "bad": "Tinggi (naik >20% dari awal)",
        "source": "Dari alat EIS"
    },
    "zimg": {
        "name": "🌀 Zimg (Ohm)",
        "desc": "Komponen reaktif (imajiner) dari impedansi.",
        "detail": "Menunjukkan sifat kapasitif baterai. Nilai negatif = kapasitif (normal).",
        "range": "-0.01 - 0.01",
        "good": "Negatif (kapasitif)",
        "bad": "Mendekati 0 atau positif",
        "source": "Dari alat EIS"
    }
}

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
        if msg["content"].startswith("<div"):
            st.markdown(f"""
            <div class="bot-message-wrapper">
                <div class="bot-avatar">🤖</div>
                <div class="bot-content">
                    <div class="bot-bubble" style="max-width: 100%;">{msg["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
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

# ==================== TAMPILKAN FORM INPUT SESUAI STEP ====================
current_step = st.session_state.step

# Fungsi untuk menampilkan input dengan tombol info menggunakan st.popover
def show_input_with_info(param_key, label, placeholder, range_text, input_key):
    param_info = PARAM_INFO[param_key]
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_val = st.text_input(label, placeholder=placeholder, key=input_key, label_visibility="collapsed")
    with col2:
        with st.popover("ℹ️ Info", use_container_width=True):
            st.markdown(f"**{param_info['name']}**")
            st.markdown(f"📖 {param_info['detail']}")
            st.markdown(f"📊 **Rentang:** {param_info['range']}")
            if param_info.get('good') and param_info.get('bad'):
                st.markdown(f"✅ **Nilai Baik:** {param_info['good']}")
                st.markdown(f"❌ **Nilai Buruk:** {param_info['bad']}")
            elif param_info.get('note'):
                st.markdown(f"📌 **Catatan:** {param_info['note']}")
            st.markdown(f"🔌 **Sumber:** {param_info['source']}")
    
    return user_val

# STEP 1: Aging Cycle
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
    
    user_val = show_input_with_info("cycle", "", "Contoh: 100", "0-2000", "cycle_input")
    
    if st.button("✅ Kirim", key="send_cycle", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                if 0 <= val <= 2000:
                    st.session_state.data["cycle"] = val
                    st.session_state.messages.append({"role": "user", "content": f"Aging cycle: {val:.0f}"})
                    st.session_state.step = "soc"
                    st.rerun()
                else:
                    st.error("⚠️ Masukkan angka antara 0-2000")
            except:
                st.error("⚠️ Masukkan angka yang valid")

# STEP 2: SOC
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
    
    user_val = show_input_with_info("soc", "", "Contoh: 80", "0-100", "soc_input")
    
    if st.button("✅ Kirim", key="send_soc", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                if 0 <= val <= 100:
                    st.session_state.data["soc"] = val
                    st.session_state.messages.append({"role": "user", "content": f"SOC: {val:.0f}%"})
                    st.session_state.step = "rint"
                    st.rerun()
                else:
                    st.error("⚠️ Masukkan angka antara 0-100")
            except:
                st.error("⚠️ Masukkan angka yang valid")

# STEP 3: R_int
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
    
    user_val = show_input_with_info("rint", "", "Contoh: 100", "0-200", "rint_input")
    
    if st.button("✅ Kirim", key="send_rint", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                if 0 <= val <= 200:
                    st.session_state.data["rint"] = val
                    st.session_state.messages.append({"role": "user", "content": f"R_int: {val:.0f}%"})
                    st.session_state.step = "ocv"
                    st.rerun()
                else:
                    st.error("⚠️ Masukkan angka antara 0-200")
            except:
                st.error("⚠️ Masukkan angka yang valid")

# STEP 4: OCV
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
    
    user_val = show_input_with_info("ocv", "", "Contoh: 4.15", "3.0-4.5", "ocv_input")
    
    if st.button("✅ Kirim", key="send_ocv", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                if 3.0 <= val <= 4.5:
                    st.session_state.data["ocv"] = val
                    st.session_state.messages.append({"role": "user", "content": f"OCV: {val:.2f}V"})
                    st.session_state.step = "freq"
                    st.rerun()
                else:
                    st.error("⚠️ Masukkan angka antara 3.0-4.5")
            except:
                st.error("⚠️ Masukkan angka yang valid")

# STEP 5: Frequency
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
    
    user_val = show_input_with_info("freq", "", "Contoh: 10", "0.1-10000", "freq_input")
    
    if st.button("✅ Kirim", key="send_freq", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                if 0.1 <= val <= 10000:
                    st.session_state.data["freq"] = val
                    st.session_state.messages.append({"role": "user", "content": f"Frequency: {val:.2f} Hz"})
                    st.session_state.step = "zmod"
                    st.rerun()
                else:
                    st.error("⚠️ Masukkan angka antara 0.1-10000")
            except:
                st.error("⚠️ Masukkan angka yang valid")

# STEP 6: Zmod
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
    
    user_val = show_input_with_info("zmod", "", "Contoh: 0.012", "0.005-0.05", "zmod_input")
    
    if st.button("✅ Kirim", key="send_zmod", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                st.session_state.data["zmod"] = val
                st.session_state.messages.append({"role": "user", "content": f"Zmod: {val:.6f} Ohm"})
                st.session_state.step = "zphz"
                st.rerun()
            except:
                st.error("⚠️ Masukkan angka yang valid")

# STEP 7: Zphz
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
    
    user_val = show_input_with_info("zphz", "", "Contoh: -2.5", "-90 - 90", "zphz_input")
    
    if st.button("✅ Kirim", key="send_zphz", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                st.session_state.data["zphz"] = val
                st.session_state.messages.append({"role": "user", "content": f"Zphz: {val:.2f}°"})
                st.session_state.step = "zreal"
                st.rerun()
            except:
                st.error("⚠️ Masukkan angka yang valid")

# STEP 8: Zreal
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
    
    user_val = show_input_with_info("zreal", "", "Contoh: 0.012", "0.005-0.02", "zreal_input")
    
    if st.button("✅ Kirim", key="send_zreal", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                st.session_state.data["zreal"] = val
                st.session_state.messages.append({"role": "user", "content": f"Zreal: {val:.6f} Ohm"})
                st.session_state.step = "zimg"
                st.rerun()
            except:
                st.error("⚠️ Masukkan angka yang valid")

# STEP 9: Zimg (terakhir, langsung prediksi)
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
    
    user_val = show_input_with_info("zimg", "", "Contoh: -0.002", "-0.01 - 0.01", "zimg_input")
    
    if st.button("✅ Kirim", key="send_zimg", use_container_width=True):
        if user_val:
            try:
                val = float(user_val)
                st.session_state.data["zimg"] = val
                st.session_state.messages.append({"role": "user", "content": f"Zimg: {val:.6f} Ohm"})
                
                # ==================== PREDIKSI SOH ====================
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
                pred_scaled = model.predict(input_scaled)
                soh = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
                
                if soh >= 90:
                    status = "SEHAT"
                    status_class = "healthy"
                    msg = "✅ Baterai dalam kondisi sangat baik! Lanjutkan pemakaian normal."
                elif soh >= 70:
                    status = "WASPADA"
                    status_class = "warning"
                    msg = "⚠️ Baterai mulai menunjukkan degradasi. Segera lakukan inspeksi."
                else:
                    status = "KRITIS"
                    status_class = "critical"
                    msg = "🔴 Kesehatan baterai kritis! Segera ganti baterai."
                
                result_html = f"""
                <div class="result-card">
                    <div class="soh-value">{soh:.1f}%</div>
                    <div class="status-badge {status_class}">{status}</div>
                    <p style="color: #ccc; margin-top: 12px;">{msg}</p>
                    <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #3E3E3E;">
                        <p style="font-size: 0.7rem; color: #666; margin: 0;">
                        📊 Berdasarkan 9 parameter yang dimasukkan
                        </p>
                    </div>
                </div>
                """
                
                st.session_state.messages.append({"role": "bot", "content": result_html})
                st.session_state.messages.append({"role": "bot", "content": "🔄 Mau cek baterai lain? Klik **Mulai Baru** di sidebar."})
                st.session_state.step = "done"
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🤖 Battery Assistant")
    st.markdown("---")
    st.markdown("### 📋 Tentang")
    st.info("""
    **Metode:** ANN (MLPRegressor)
    
    **Dataset:** CNR Italy EIS (8 Cells, 2026)
    
    **9 Parameter Input:**
    - 🔄 Aging Cycle
    - 🔋 SOC (%)
    - ⚡ R_int (%)
    - 🔌 OCV (V)
    - 📊 Frequency (Hz)
    - 📈 Zmod (Ohm)
    - 🔄 Zphz (deg)
    - 📉 Zreal (Ohm)
    - 🌀 Zimg (Ohm)
    
    **Output:** SOH (%) + Status
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Status SOH")
    st.markdown("""
    - 🟢 **≥90%** → SEHAT
    - 🟡 **70-90%** → WASPADA
    - 🔴 **<70%** → KRITIS
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
    🔋 Battery Assistant — Prediksi SOH Baterai dengan ANN (9 Parameter)
</div>
""", unsafe_allow_html=True)
