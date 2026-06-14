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
    
    /* Bot message */
    .bot-message {
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
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
    }
    
    .bot-bubble {
        background: #1E1E1E;
        border-radius: 20px 20px 20px 4px;
        padding: 16px 20px;
        color: #E0E0E0;
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 80%;
        border: 1px solid #2E2E2E;
    }
    
    /* User message */
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
    
    /* Parameter explanation card */
    .param-card {
        background: #1A1A1A;
        border-radius: 14px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 4px solid;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .param-card:hover {
        background: #222222;
    }
    
    .param-name {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    
    .param-desc {
        font-size: 0.75rem;
        color: #888;
        margin-bottom: 6px;
    }
    
    .param-range {
        font-size: 0.7rem;
        color: #10B981;
        font-family: monospace;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1E1E1E, #2E2E2E);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        margin: 16px 0;
        border: 1px solid #3E3E3E;
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
    
    /* Input area */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #0D0D0D;
        padding: 1rem;
        border-top: 1px solid #2E2E2E;
    }
    
    .input-wrapper {
        max-width: 900px;
        margin: 0 auto;
        display: flex;
        gap: 12px;
    }
    
    .input-wrapper input {
        flex: 1;
        background: #1E1E1E;
        border: 1px solid #2E2E2E;
        border-radius: 30px;
        padding: 14px 20px;
        color: white;
        font-size: 0.95rem;
    }
    
    .input-wrapper input:focus {
        outline: none;
        border-color: #10B981;
    }
    
    .input-wrapper button {
        background: linear-gradient(135deg, #10B981, #06B6D4);
        border: none;
        border-radius: 30px;
        padding: 14px 28px;
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .input-wrapper button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .typing {
        display: inline-block;
        width: 4px;
        height: 4px;
        background: #10B981;
        border-radius: 50%;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.7rem;
        padding: 20px;
        margin-top: 100px;
    }
    
    .divider {
        border: none;
        border-top: 1px solid #2E2E2E;
        margin: 20px 0;
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

model, scaler_X, scaler_y = load_model()

# ==================== SESSION STATE ====================
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.data = {}
    st.session_state.messages = [
        {"role": "bot", "content": "👋 Halo! Aku **Battery Assistant** 🤖\n\nAku akan membantu memprediksi kesehatan baterai mobil listrikmu.\n\nSebelum mulai, aku akan jelaskan dulu parameter yang perlu kamu masukkan:"},
        {"role": "bot", "content": """
**📋 PARAMETER YANG DIBUTUHKAN:**

1. **🔄 Aging Cycle** - Jumlah siklus charge-discharge baterai (0-2000)
2. **🔋 SOC (%)** - Level pengisian baterai saat ini (0-100%)
3. **⚡ R_int (%)** - Hambatan internal baterai (100% = normal)
4. **🔌 OCV (V)** - Tegangan baterai saat diam (3.0-4.5V)
5. **📊 Frequency (Hz)** - Frekuensi pengukuran EIS (0.1-10000 Hz)
6. **📈 Zmod (Ohm)** - Impedansi modulus
7. **🔄 Zphz (deg)** - Sudut fase impedansi
8. **📉 Zreal (Ohm)** - Resistansi nyata
9. **🌀 Zimg (Ohm)** - Reaktansi imajiner

Siap? Yuk mulai! Masukkan **Aging Cycle** (0-2000): 
"""}
    ]
    st.session_state.current_input = "cycle"

# ==================== PARAMETER EXPLANATIONS ====================
PARAM_INFO = {
    "cycle": {
        "name": "🔄 Aging Cycle",
        "desc": "Jumlah siklus charge-discharge yang sudah dilalui baterai. 1 siklus = charge 0% → 100%. Semakin tinggi, baterai semakin aus.",
        "normal": "0 - 200 cycle (baru), >600 cycle (mulai menua)",
        "good": "Rendah (<300)",
        "bad": "Tinggi (>600)"
    },
    "soc": {
        "name": "🔋 SOC (%)",
        "desc": "State of Charge — level pengisian baterai saat ini. Idealnya 20-80% untuk menjaga kesehatan baterai.",
        "normal": "20% - 80%",
        "good": "20-80%",
        "bad": "<20% atau >80%"
    },
    "rint": {
        "name": "⚡ R_int (%)",
        "desc": "Internal Resistance — hambatan listrik di dalam baterai. Semakin tinggi, baterai semakin rusak.",
        "normal": "100% = normal",
        "good": "≤110%",
        "bad": ">150%"
    },
    "ocv": {
        "name": "🔌 OCV (V)",
        "desc": "Open Circuit Voltage — tegangan baterai saat tidak dipakai. Baterai sehat di atas 3.8V.",
        "normal": "3.8V - 4.2V",
        "good": "≥3.9V",
        "bad": "<3.7V"
    },
    "freq": {
        "name": "📊 Frequency (Hz)",
        "desc": "Frekuensi pengukuran EIS. Frekuensi rendah (0.1Hz) sensitif ke degradasi, tinggi (1000Hz) ke resistansi murni.",
        "normal": "0.1 - 10000 Hz",
        "good": "-",
        "bad": "-"
    },
    "zmod": {
        "name": "📈 Zmod (Ohm)",
        "desc": "Modulus impedansi — besaran impedansi total baterai. Nilai meningkat seiring degradasi.",
        "normal": "0.01 - 0.02 Ohm",
        "good": "Rendah",
        "bad": "Tinggi"
    },
    "zphz": {
        "name": "🔄 Zphz (deg)",
        "desc": "Sudut fase impedansi. Menunjukkan sifat baterai (resistif/kapasitif).",
        "normal": "-10° - 0°",
        "good": "-",
        "bad": "-"
    },
    "zreal": {
        "name": "📉 Zreal (Ohm)",
        "desc": "Komponen resistif (nyata) dari impedansi. Berkorelasi langsung dengan R_int.",
        "normal": "0.01 - 0.02 Ohm",
        "good": "Rendah",
        "bad": "Tinggi"
    },
    "zimg": {
        "name": "🌀 Zimg (Ohm)",
        "desc": "Komponen reaktif (imajiner) dari impedansi. Menunjukkan sifat kapasitif baterai.",
        "normal": "-0.005 - 0.005 Ohm",
        "good": "-",
        "bad": "-"
    }
}

# ==================== FUNGSI PROSES INPUT ====================
def process_input(user_input):
    step = st.session_state.current_input
    
    if step == "cycle":
        try:
            val = float(user_input)
            if 0 <= val <= 2000:
                st.session_state.data["cycle"] = val
                st.session_state.current_input = "soc"
                return f"✅ Aging cycle: {val:.0f}\n\nSekarang masukkan **SOC (%)** (State of Charge, 0-100):\n\n📌 *{PARAM_INFO['soc']['desc'][:100]}...*"
            return "⚠️ Masukkan angka antara 0-2000:"
        except:
            return "⚠️ Masukkan angka yang valid:"
    
    elif step == "soc":
        try:
            val = float(user_input)
            if 0 <= val <= 100:
                st.session_state.data["soc"] = val
                st.session_state.current_input = "rint"
                tip = "✅ Optimal!" if 20 <= val <= 80 else ("⚠️ Terlalu rendah!" if val < 20 else "⚠️ Terlalu tinggi!")
                return f"✅ SOC: {val:.0f}% {tip}\n\nSekarang masukkan **R_int (%)** (Internal Resistance, normal 100%):\n\n📌 *{PARAM_INFO['rint']['desc'][:100]}...*"
            return "⚠️ Masukkan angka 0-100:"
        except:
            return "⚠️ Masukkan angka yang valid:"
    
    elif step == "rint":
        try:
            val = float(user_input)
            if 0 <= val <= 200:
                st.session_state.data["rint"] = val
                st.session_state.current_input = "ocv"
                tip = "✅ Normal" if val <= 110 else ("⚠️ Mulai tinggi" if val <= 150 else "🔴 Sangat tinggi")
                return f"✅ R_int: {val:.0f}% ({tip})\n\nSekarang masukkan **OCV (V)** (Open Circuit Voltage, 3.0-4.5V):\n\n📌 *{PARAM_INFO['ocv']['desc'][:100]}...*"
            return "⚠️ Masukkan angka 0-200:"
        except:
            return "⚠️ Masukkan angka yang valid:"
    
    elif step == "ocv":
        try:
            val = float(user_input)
            if 3.0 <= val <= 4.5:
                st.session_state.data["ocv"] = val
                st.session_state.current_input = "freq"
                tip = "✅ Normal" if val >= 3.9 else ("⚠️ Mulai turun" if val >= 3.7 else "🔴 Sangat rendah")
                return f"✅ OCV: {val:.2f}V ({tip})\n\nSekarang masukkan **Frequency (Hz)** (0.1-10000 Hz):\n\n📌 *Frekuensi pengukuran EIS. Frekuensi rendah (0.1Hz) sensitif ke degradasi.*"
            return "⚠️ Masukkan angka 3.0-4.5:"
        except:
            return "⚠️ Masukkan angka yang valid:"
    
    elif step == "freq":
        try:
            val = float(user_input)
            if 0.1 <= val <= 10000:
                st.session_state.data["freq"] = val
                st.session_state.current_input = "zmod"
                return f"✅ Frequency: {val:.2f} Hz\n\nSekarang masukkan **Zmod (Ohm)** (Impedance modulus):\n\n📌 *Besaran impedansi total baterai. Biasanya 0.01-0.02 Ohm.*"
            return "⚠️ Masukkan angka 0.1-10000:"
        except:
            return "⚠️ Masukkan angka yang valid:"
    
    elif step == "zmod":
        try:
            val = float(user_input)
            st.session_state.data["zmod"] = val
            st.session_state.current_input = "zphz"
            return f"✅ Zmod: {val:.6f} Ohm\n\nSekarang masukkan **Zphz (deg)** (Sudut fase impedansi):\n\n📌 *Sudut fase impedansi. Biasanya -10° sampai 10°.*"
        except:
            return "⚠️ Masukkan angka yang valid:"
    
    elif step == "zphz":
        try:
            val = float(user_input)
            st.session_state.data["zphz"] = val
            st.session_state.current_input = "zreal"
            return f"✅ Zphz: {val:.2f}°\n\nSekarang masukkan **Zreal (Ohm)** (Resistansi nyata):\n\n📌 *Komponen resistif impedansi. Biasanya 0.01-0.02 Ohm.*"
        except:
            return "⚠️ Masukkan angka yang valid:"
    
    elif step == "zreal":
        try:
            val = float(user_input)
            st.session_state.data["zreal"] = val
            st.session_state.current_input = "zimg"
            return f"✅ Zreal: {val:.6f} Ohm\n\nTerakhir, masukkan **Zimg (Ohm)** (Reaktansi imajiner):\n\n📌 *Komponen reaktif impedansi. Biasanya -0.005 sampai 0.005 Ohm.*"
        except:
            return "⚠️ Masukkan angka yang valid:"
    
    elif step == "zimg":
        try:
            val = float(user_input)
            st.session_state.data["zimg"] = val
            st.session_state.current_input = "done"
            
            # Prediksi SOH
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
                    <p style="font-size: 0.75rem; color: #888; margin: 0;">
                    🔍 Berdasarkan parameter yang dimasukkan, model ANN memprediksi SOH = {soh:.1f}%
                    </p>
                </div>
            </div>
            """
            
            return result_html
        
        except Exception as e:
            return f"⚠️ Error: {e}"
    
    return "Terima kasih!"

# ==================== UI HEADER ====================
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
        <div class="bot-message">
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

# ==================== INPUT FORM ====================
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", placeholder="Ketik jawabanmu di sini...", label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("📤 Kirim")

if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    bot_response = process_input(user_input)
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    
    if st.session_state.current_input == "done":
        st.session_state.messages.append({"role": "bot", "content": "🔄 Mau cek baterai lain? Klik **Mulai Baru** di sidebar."})
        st.session_state.current_input = "cycle"
        st.session_state.data = {}
    
    st.rerun()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## 🤖 Battery Assistant")
    st.markdown("---")
    st.markdown("### 📋 Tentang")
    st.info("""
    **Metode:** ANN (MLPRegressor)
    
    **Dataset:** CNR Italy EIS (8 Cells, 2026)
    
    **9 Parameter Input:**
    - Aging Cycle (0-2000)
    - SOC (%) (0-100)
    - R_int (%) (0-200)
    - OCV (V) (3.0-4.5)
    - Frequency (Hz)
    - Zmod (Ohm)
    - Zphz (deg)
    - Zreal (Ohm)
    - Zimg (Ohm)
    
    **Output:** SOH (%) + Status (Sehat/Waspada/Kritis)
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
        st.session_state.step = 0
        st.session_state.data = {}
        st.session_state.messages = [
            {"role": "bot", "content": "👋 Halo! Aku **Battery Assistant** 🤖\n\nAku akan membantu memprediksi kesehatan baterai mobil listrikmu.\n\nSiap? Yuk mulai! Masukkan **Aging Cycle** (0-2000):"}
        ]
        st.session_state.current_input = "cycle"
        st.rerun()
    
    st.caption("© 2026 | Project SC 2026")

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
    🔋 Battery Assistant — Prediksi SOH Baterai dengan ANN (9 Parameter)
</div>
""", unsafe_allow_html=True)
