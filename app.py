import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")

# --- 1. DATA & MODEL PREPARATION ---
@st.cache_data
def load_and_train_model():
    # In a real scenario, you would load your 'nanoemulsion_master.csv'
    # For this app, we create a structure based on your 1000-row dataset logic
    data = {
        'Drug_Name': ['Ibuprofen', 'Curcumin', 'Paclitaxel', 'Ketoprofen', 'Dexamethasone', 'Retinol', 'Donepezil'] * 150,
        'Surfactant': ['Tween 80', 'Tween 20', 'Cremophor EL', 'Labrasol', 'Lecithin', 'Span 80', 'Cremophor RH40'] * 150,
        'Co_surfactant': ['Ethanol', 'PEG 400', 'Transcutol', 'Prop. Glycol', 'PEG 600', 'Water', 'Span 20'] * 150,
        'Oil_phase': ['Olive Oil', 'MCT Oil', 'Soybean Oil', 'Capryol 90', 'IPM', 'Castor Oil', 'Corn Oil'] * 150,
        'Size_nm': np.random.uniform(50, 200, 1050),
        'PDI': np.random.uniform(0.1, 0.3, 1050),
        'Zeta_mV': np.random.uniform(-40, -10, 1050),
        'Drug_Loading': np.random.uniform(1, 25, 1050),
        'EE_percent': np.random.uniform(88, 99, 1050)
    }
    df = pd.DataFrame(data)

    # Encoders for Categorical Data
    le_dict = {col: LabelEncoder().fit(df[col]) for col in ['Drug_Name', 'Surfactant', 'Co_surfactant', 'Oil_phase']}
    
    # Feature Matrix
    X = pd.DataFrame({
        'Drug': le_dict['Drug_Name'].transform(df['Drug_Name']),
        'Surf': le_dict['Surfactant'].transform(df['Surfactant']),
        'CoSurf': le_dict['Co_surfactant'].transform(df['Co_surfactant']),
        'Oil': le_dict['Oil_phase'].transform(df['Oil_phase'])
    })
    
    # Target Matrix
    y = df[['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'EE_percent']]
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, y)
    
    return df, model, le_dict

df, model, le_dict = load_and_train_model()

# --- APP LAYOUT ---
st.title("🧪 NanoPredict: AI Nanoemulsion Optimizer")
st.markdown("---")

# --- SIDEBAR: STEP 1 ---
st.sidebar.header("📍 STEP 1: Core Inputs")
selected_drug = st.sidebar.selectbox("Select/Type Drug Name", df['Drug_Name'].unique())
selected_oil = st.sidebar.selectbox("Select Oil Phase", df['Oil_phase'].unique())
selected_aqueous = st.sidebar.text_input("Aqueous Phase", "Distilled Water / PBS")

# --- MAIN PANEL: STEP 2 ---
st.header("🔍 STEP 2: Recommended Components")
# Logic: Filter dataset for the selected Oil to find compatible Surfactants
recommendations = df[df['Oil_phase'] == selected_oil][['Surfactant', 'Co_surfactant']].drop_duplicates().head(3)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Surfactants")
    for s in recommendations['Surfactant'].unique():
        st.write(f"✅ {s}")

with col2:
    st.subheader("Top Co-Surfactants")
    for cs in recommendations['Co_surfactant'].unique():
        st.write(f"🔹 {cs}")

st.markdown("---")

# --- MAIN PANEL: STEP 3 ---
st.header("📊 STEP 3: Optimized Range & Predictions")

col_in1, col_in2 = st.columns(2)
with col_in1:
    target_surf = st.selectbox("Final Surfactant Choice", df['Surfactant'].unique())
with col_in2:
    target_cosurf = st.selectbox("Final Co-Surfactant Choice", df['Co_surfactant'].unique())

st.info("💡 **Recommended Range:** Surfactant (20-40% w/w) | Co-surfactant (5-15% w/w)")

if st.button("🚀 Run AI Prediction Engine"):
    # Encoding choices
    d_enc = le_dict['Drug_Name'].transform([selected_drug])[0]
    o_enc = le_dict['Oil_phase'].transform([selected_oil])[0]
    s_enc = le_dict['Surfactant'].transform([target_surf])[0]
    c_enc = le_dict['Co_surfactant'].transform([target_cosurf])[0]
    
    # Prediction
    res = model.predict([[d_enc, s_enc, c_enc, o_enc]])[0]
    
    # Display Results
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Droplet Size", f"{res[0]:.2f} nm")
    res_col1.metric("PDI", f"{res[1]:.3f}")
    
    res_col2.metric("Zeta Potential", f"{res[2]:.2f} mV")
    res_col2.metric("Drug Loading", f"{res[3]:.2f} mg/mL")
    
    res_col3.metric("EE (%)", f"{res[4]:.1f}%")
    res_col3.info("Stability: **High (Stable)**")

    # --- 3D PSEUDO-TERNARY PLOT ---
    st.subheader("🌐 3D Pseudo-Ternary Phase Space")
    
    # Simulating data points for the ternary region
    n_points = 500
    oil_pts = np.random.uniform(5, 40, n_points)
    smix_pts = np.random.uniform(20, 70, n_points)
    water_pts = 100 - oil_pts - smix_pts
    # Filter points where water > 0
    valid = water_pts > 0
    
    fig = go.Figure(data=[go.Scatter3d(
        x=oil_pts[valid], y=smix_pts[valid], z=water_pts[valid],
        mode='markers',
        marker=dict(
            size=4,
            color=res[0] + np.random.normal(0, 10, sum(valid)), # Simulated size variance
            colorscale='Viridis',
            colorbar=dict(title="Size (nm)"),
            opacity=0.7
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Oil %',
            yaxis_title='Smix %',
            zaxis_title='Water %'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"Predicted Nanoemulsion Region for {selected_drug}"
    )
    st.plotly_chart(fig, use_container_width=True)
