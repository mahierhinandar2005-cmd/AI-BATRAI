import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Battery Assistant", page_icon="🔋", layout="wide")

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

# ==================== HEADER ====================
st.title("🔋 Battery Assistant")
st.markdown("Prediksi State of Health (SOH) Baterai Mobil Listrik")
st.markdown("---")

# ==================== INPUT FORM (SEMUA PARAMETER SEKALIGUS) ====================
with st.form("battery_form"):
    st.subheader("📝 Input Parameter Baterai")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cycle = st.number_input("Aging Cycle", 0, 2000, 100)
        soc = st.number_input("SOC (%)", 0, 100, 80)
        rint = st.number_input("R_int (%)", 0, 200, 100)
        ocv = st.number_input("OCV (V)", 3.0, 4.5, 4.15, 0.05)
    
    with col2:
        freq = st.number_input("Frequency (Hz)", 0.1, 10000.0, 10.0)
        zmod = st.number_input("Zmod (Ohm)", 0.0, 0.1, 0.012, format="%.6f")
        zphz = st.number_input("Zphz (deg)", -90.0, 90.0, -2.5)
        zreal = st.number_input("Zreal (Ohm)", 0.0, 0.1, 0.012, format="%.6f")
        zimg = st.number_input("Zimg (Ohm)", -0.1, 0.1, -0.002, format="%.6f")
    
    submitted = st.form_submit_button("🔮 PREDIKSI", type="primary", use_container_width=True)

# ==================== FUNGSI ANALISIS ====================
def get_status(name, value):
    if name == "Aging Cycle":
        if value < 300:
            return "good", f"{value} (Masih baru)", "Lanjutkan normal"
        elif value < 600:
            return "service", f"{value} (Mulai menua)", "Pantau berkala"
        else:
            return "replace", f"{value} (Sudah tinggi)", "Inspeksi segera"
    elif name == "SOC (%)":
        if 20 <= value <= 80:
            return "good", f"{value}% (Optimal)", "Kebiasaan baik"
        else:
            return "service", f"{value}% (Tidak ideal)", "Sesuaikan charge"
    elif name == "R_int (%)":
        if value <= 110:
            return "good", f"{value}% (Normal)", "Pertahankan"
        elif value <= 150:
            return "service", f"{value}% (Mulai naik)", "Balancing cell"
        else:
            return "replace", f"{value}% (Sangat tinggi)", "Ganti baterai"
    elif name == "OCV (V)":
        if value >= 3.9:
            return "good", f"{value}V (Normal)", "Tidak perlu tindakan"
        elif value >= 3.7:
            return "service", f"{value}V (Mulai turun)", "Periksa sistem"
        else:
            return "replace", f"{value}V (Sangat rendah)", "Ganti baterai"
    elif name in ["Zmod (Ohm)", "Zreal (Ohm)"]:
        if value <= 0.015:
            return "good", f"{value} (Normal)", "Lanjutkan"
        elif value <= 0.025:
            return "service", f"{value} (Mulai naik)", "Monitor"
        else:
            return "replace", f"{value} (Tinggi)", "Indikasi rusak"
    elif name == "Zphz (deg)":
        if value <= -5:
            return "good", f"{value}° (Normal)", "Lanjutkan"
        elif value <= 0:
            return "service", f"{value}° (Mulai berubah)", "Periksa"
        else:
            return "replace", f"{value}° (Positif)", "Indikasi rusak"
    elif name == "Zimg (Ohm)":
        if value < -0.002:
            return "good", f"{value} (Normal)", "Lanjutkan"
        elif value <= 0:
            return "service", f"{value} (Mulai berubah)", "Periksa"
        else:
            return "replace", f"{value} (Positif)", "Indikasi rusak"
    return "good", str(value), "Normal"

# ==================== PROSES PREDIKSI ====================
if submitted and model_loaded:
    # Prediksi SOH
    input_data = np.array([[cycle, soc, rint, ocv, freq, zmod, zphz, zreal, zimg]])
    input_scaled = scaler_X.transform(input_data)
    soh = scaler_y.inverse_transform(model.predict(input_scaled).reshape(-1, 1))[0][0]
    
    # Status SOH
    if soh >= 90:
        status = "SEHAT"
        color = "green"
    elif soh >= 70:
        status = "WASPADA"
        color = "orange"
    else:
        status = "KRITIS"
        color = "red"
    
    st.markdown("---")
    st.subheader("📊 HASIL PREDIKSI")
    
    # Tampilkan SOH
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: #1E1E1E; border-radius: 20px;">
            <h1 style="font-size: 3.5rem; color: {color};">{soh:.1f}%</h1>
            <h3 style="color: {color};">{status}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        title={"text": "SOH Meter"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 70], "color": "#FF6B6B"},
                {"range": [70, 90], "color": "#FFD93D"},
                {"range": [90, 100], "color": "#6BCB77"}
            ]
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analisis per parameter
    st.markdown("---")
    st.subheader("🔍 Analisis Kondisi Setiap Parameter")
    
    params = [
        ("Aging Cycle", cycle),
        ("SOC (%)", soc),
        ("R_int (%)", rint),
        ("OCV (V)", ocv),
        ("Zmod (Ohm)", zmod),
        ("Zphz (deg)", zphz),
        ("Zreal (Ohm)", zreal),
        ("Zimg (Ohm)", zimg)
    ]
    
    good_count = 0
    service_count = 0
    replace_count = 0
    
    for name, value in params:
        stat, msg, action = get_status(name, value)
        if stat == "good":
            good_count += 1
            st.success(f"✅ **{name}** : {msg} → {action}")
        elif stat == "service":
            service_count += 1
            st.warning(f"⚠️ **{name}** : {msg} → {action}")
        else:
            replace_count += 1
            st.error(f"🔴 **{name}** : {msg} → {action}")
    
    # Ringkasan
    st.markdown("---")
    st.subheader("📋 RINGKASAN REKOMENDASI")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ LANJUTKAN", f"{good_count} parameter")
    with col2:
        st.metric("⚠️ PERLU SERVICE", f"{service_count} parameter")
    with col3:
        st.metric("🔴 WAJIB GANTI", f"{replace_count} parameter")
    
    # Kesimpulan
    st.markdown("---")
    if replace_count > 0:
        st.error(f"🔴 **KESIMPULAN:** Ada {replace_count} parameter yang wajib GANTI. Segera lakukan tindakan!")
    elif service_count > 0:
        st.warning(f"⚠️ **KESIMPULAN:** Ada {service_count} parameter yang perlu SERVICE. Lakukan inspeksi segera.")
    else:
        st.success(f"✅ **KESIMPULAN:** Semua parameter dalam kondisi baik. Lanjutkan pemakaian normal.")

elif submitted and not model_loaded:
    st.error("Model tidak tersedia")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 📋 Panduan Parameter")
    st.markdown("""
    | Parameter | Normal | Service | Ganti |
    |:---|:---|:---|:---|
    | Aging Cycle | <300 | 300-600 | >600 |
    | SOC (%) | 20-80 | di luar | - |
    | R_int (%) | ≤110 | 110-150 | >150 |
    | OCV (V) | ≥3.9 | 3.7-3.9 | <3.7 |
    | Zmod/Zreal | ≤0.015 | 0.015-0.025 | >0.025 |
    | Zphz | ≤-5 | -5-0 | >0 |
    | Zimg | <-0.002 | -0.002-0 | >0 |
    """)
    st.markdown("---")
    st.caption("© 2026 | Project SC 2026")
