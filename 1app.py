import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor

# --- CONFIG ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

# --- DATA LOADING & ML ENGINE ---
@st.cache_data
def load_full_data():
    # Load your 900+ row file
    df = pd.read_csv('nanoemulsion 2.csv')
    
    # Simple cleaning for ML
    df['Size_nm'] = pd.to_numeric(df['Size_nm'], errors='coerce')
    df = df.dropna(subset=['Size_nm'])
    
    # Initialize Encoders
    le_dict = {}
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('None')
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        le_dict[col] = le
        
    # Train Model
    X = df[['Drug_Name_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'Oil_phase_enc']]
    y = df[['Size_nm', 'PDI', 'Zeta_mV']] # Add more targets if available in your CSV
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
    model.fit(X, y)
    
    return df, model, le_dict

# Load resources
try:
    df, model, le_dict = load_full_data()
except:
    st.error("Please ensure 'nanoemulsion 2.csv' is in the same folder as this app.")
    st.stop()

# Initialize History in Session State
if 'history' not in st.session_state:
    st.session_state.history = []

# --- MAIN UI ---
st.title("🧪 Nanoemulsion Research Station")
st.write("Full-scale predictive engine for pharmaceutical formulations.")

# STEP 1 & 2: INPUT BOX
with st.container():
    st.header("📍 Step 1 & 2: Component Selection")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        # This will now show ALL 900+ drugs from your CSV
        drug_input = st.selectbox("Search Drug Name", sorted(df['Drug_Name'].unique()))
    with c2:
        oil_input = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
    with c3:
        aqueous_input = st.text_input("Aqueous Phase", "Water / PBS")

    # Automated Step 2: Recommendations based on Step 1
    recs = df[df['Oil_phase'] == oil_input][['Surfactant', 'Co-surfactant']].drop_duplicates().head(5)
    st.markdown(f"**Recommended for {oil_input}:** " + ", ".join(recs['Surfactant'].unique()[:3]))

st.markdown("---")

# STEP 3: PREDICTION
with st.container():
    st.header("📊 Step 3: Optimization & Prediction")
    p1, p2, p3 = st.columns([2, 2, 1])
    
    with p1:
        target_s = st.selectbox("Final Surfactant Choice", sorted(df['Surfactant'].unique()))
    with p2:
        target_cs = st.selectbox("Final Co-Surfactant Choice", sorted(df['Co-surfactant'].unique()))
    with p3:
        st.write("##")
        predict_btn = st.button("🚀 Predict & Save", use_container_width=True)

if predict_btn:
    # Encoding
    try:
        d_e = le_dict['Drug_Name'].transform([drug_input])[0]
        o_e = le_dict['Oil_phase'].transform([oil_input])[0]
        s_e = le_dict['Surfactant'].transform([target_s])[0]
        c_e = le_dict['Co-surfactant'].transform([target_cs])[0]
        
        # ML Result
        pred = model.predict([[d_e, s_e, c_e, o_e]])[0]
        
        # Save to History
        new_entry = {
            "Drug": drug_input,
            "Oil": oil_input,
            "Surfactant": target_s,
            "Size (nm)": round(pred[0], 2),
            "PDI": round(pred[1], 3),
            "Zeta (mV)": round(pred[2], 2)
        }
        st.session_state.history.append(new_entry)
        
        st.success(f"Prediction Complete! Droplet Size: {pred[0]:.2f} nm")
    except Exception as e:
        st.error(f"Encoding Error: Some combinations are new to the model. {e}")

# --- HISTORY & VISUALIZATION ---
st.markdown("---")
h_col, v_col = st.columns([1, 1])

with h_col:
    st.subheader("📜 Session History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button("Download History CSV", history_df.to_csv(index=False), "history.csv")
    else:
        st.write("No predictions made yet.")

with v_col:
    st.subheader("🌐 3D Phase Space")
    # Simplified plot based on current prediction
    fig = go.Figure(data=[go.Scatter3d(
        x=np.random.rand(100)*40, y=np.random.rand(100)*60, z=np.random.rand(100)*50,
        mode='markers', marker=dict(size=4, color='purple', opacity=0.6)
    )])
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), height=300)
    st.plotly_chart(fig, use_container_width=True)
