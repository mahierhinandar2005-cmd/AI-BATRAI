import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Battery Assistant", page_icon="🔋", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background: #0D0D0D; }
    .bot-bubble { background: #1E1E1E; border-radius: 18px; padding: 12px 18px; color: #E0E0E0; }
    .user-bubble { background: linear-gradient(135deg, #10B981, #06B6D4); border-radius: 18px; padding: 12px 18px; color: white; }
    .result-card { background: linear-gradient(135deg, #1E1E1E, #2E2E2E); border-radius: 16px; padding: 12px; text-align: center; margin: 8px 0; }
    .soh-value { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #10B981, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .healthy { background: rgba(16, 185, 129, 0.2); color: #10B981; }
    .warning { background: rgba(245, 158, 11, 0.2); color: #F59E0B; }
    .critical { background: rgba(239, 68, 68, 0.2); color: #EF4444; }
    .input-area { position: fixed; bottom: 0; left: 0; right: 0; background: #0D0D0D; padding: 1rem; border-top: 1px solid #2E2E2E; }
    .footer { text-align: center; color: #666; font-size: 0.7rem; padding: 20px; margin-top: 80px; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('model_soh.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model()

# Header
st.markdown('<div style="text-align: center;"><h1 style="color: white;">🔋 Battery Assistant</h1><p style="color: #888;">Prediksi SOH Baterai dengan ANN (9 Parameter)</p></div>', unsafe_allow_html=True)

# Input form
st.markdown("### 📝 Input Parameter Baterai")

col1, col2 = st.columns(2)

with col1:
    cycle = st.number_input("🔄 Aging Cycle", min_value=0, max_value=2000, value=100, step=10)
    soc = st.number_input("🔋 SOC (%)", min_value=0, max_value=100, value=80, step=5)
    r_int = st.number_input("⚡ R_int (%)", min_value=0, max_value=200, value=100, step=5)
    ocv = st.number_input("🔌 OCV (V)", min_value=3.0, max_value=4.5, value=4.15, step=0.05)
    freq = st.number_input("📊 Frequency (Hz)", min_value=0.1, max_value=10000.0, value=10.0, step=1.0)

with col2:
    zmod = st.number_input("📈 Zmod (Ohm)", min_value=0.0, max_value=1.0, value=0.012, step=0.001, format="%.6f")
    zphz = st.number_input("🔄 Zphz (deg)", min_value=-90.0, max_value=90.0, value=0.0, step=1.0)
    zreal = st.number_input("📉 Zreal (Ohm)", min_value=0.0, max_value=1.0, value=0.012, step=0.001, format="%.6f")
    zimg = st.number_input("🌀 Zimg (Ohm)", min_value=-0.5, max_value=0.5, value=0.0, step=0.001, format="%.6f")

if st.button("🔮 PREDIKSI KESEHATAN BATERAI", type="primary", use_container_width=True):
    input_data = np.array([[cycle, soc, r_int, ocv, freq, zmod, zphz, zreal, zimg]])
    input_scaled = scaler_X.transform(input_data)
    pred_scaled = model.predict(input_scaled)
    soh = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    
    if soh >= 90:
        status = "SEHAT"
        status_class = "healthy"
    elif soh >= 70:
        status = "WASPADA"
        status_class = "warning"
    else:
        status = "KRITIS"
        status_class = "critical"
    
    st.markdown(f"""
    <div class="result-card">
        <div class="soh-value">{soh:.1f}%</div>
        <div><span class="status-badge {status_class}">{status}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh,
        title={"text": "SOH Meter"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#10B981"},
            "steps": [
                {"range": [0, 70], "color": "#7F1A1A"},
                {"range": [70, 90], "color": "#854D0E"},
                {"range": [90, 100], "color": "#14532D"}
            ]
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🔋 9 Parameter: Cycle, SOC, R_int, OCV, Frequency, Zmod, Zphz, Zreal, Zimg | Model ANN")
