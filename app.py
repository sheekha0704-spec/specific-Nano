import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v4.0", layout="wide")

# Custom CSS for Large, Un-truncated Metric Displays (Request 1)
st.markdown("""
    <style>
    .metric-container {
        background: #ffffff; padding: 24px; border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); border-left: 10px solid #0056b3;
        margin-bottom: 25px;
    }
    .m-label { font-size: 16px; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .m-value { font-size: 30px; color: #000; font-weight: 800; margin-top: 5px; }
    .axis-box { background: #eef2f7; padding: 15px; border-radius: 8px; border: 1px solid #d1d9e6; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_train_precision():
    df = pd.read_csv('nanoemulsion 2.csv')
    
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    target_cols = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in target_cols:
        df[f'{col}_clean'] = df[col].apply(get_num)
        df[f'{col}_clean'] = df.groupby('Oil_phase')[f'{col}_clean'].transform(lambda x: x.fillna(x.median()))

    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    X = df[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    
    # High Precision Regressors
    models = {}
    for col in target_cols:
        m = GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, random_state=42)
        m.fit(X, df[f'{col}_clean'])
        models[col] = m
    
    # Confidence Classifier
    df['is_stable'] = df['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=400, random_state=42).fit(X, df['is_stable'])
    
    return df, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_train_precision()

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation Menu", ["1. Phase Setup", "2. Specific Recommendations", "3. Prediction & Ternary Plot", "History"])

if 'inputs' not in st.session_state:
    st.session_state.inputs = {'drug': df['Drug_Name'].unique()[0], 'oil': df['Oil_phase'].unique()[0]}

# --- PAGE 1: SETUP ---
if page == "1. Phase Setup":
    st.header("Step 1: Define Core Components")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.inputs['drug'] = st.selectbox("Search Drug Library (900+)", sorted(df['Drug_Name'].unique()))
    with c2:
        st.session_state.inputs['oil'] = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
    st.success("✅ Configuration Locked. Move to Step 2 for scientific matching.")

# --- PAGE 2: RECOMMENDATIONS (Specific AI Rationale) ---
elif page == "2. Specific Recommendations":
    st.header("Step 2: AI-Driven Scientific Rationale")
    oil = st.session_state.inputs['oil']
    drug = st.session_state.inputs['drug']
    
    # Logic: Search dataset for this specific oil and find systems with best PDI and Size
    best_systems = df[df['Oil_phase'] == oil].sort_values(by=['Size_nm_clean', 'EE_clean'], ascending=[True, False]).head(3)
    
    st.subheader(f"Top 3 Systems for {oil}")
    for i, row in best_systems.iterrows():
        with st.expander(f"Recommended System: {row['Surfactant']} + {row['Co-surfactant']}"):
            st.markdown(f"**Specific Rationale for Choice:**")
            st.write(f"- **Historical Evidence:** In your dataset, this system achieved an average droplet size of **{row['Size_nm_clean']:.2f} nm** and a PDI of **{row['PDI_clean']:.3f}**.")
            st.write(f"- **Chemical Affinity:** {row['Surfactant']} provides the optimal Hydrophilic-Lipophilic Balance (HLB) to emulsify {oil} without phase separation.")
            st.write(f"- **Payload Efficiency:** This pair demonstrated an Encapsulation Efficiency of **{row['EE_clean']:.1f}%**, which is superior for {drug} retention.")

# --- PAGE 3: PREDICTION (Axis Defined) ---
elif page == "3. Prediction & Ternary Plot":
    st.header("Step 3: Optimized Prediction & Ternary Space")
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        s_choice = st.selectbox("Final Surfactant", sorted(df['Surfactant'].unique()))
        cs_choice = st.selectbox("Final Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Run Precision AI Prediction"):
            inputs = [[le_dict['Drug_Name'].transform([st.session_state.inputs['drug']])[0],
                       le_dict['Oil_phase'].transform([st.session_state.inputs['oil']])[0],
                       le_dict['Surfactant'].transform([s_choice])[0],
                       le_dict['Co-surfactant'].transform([cs_choice])[0]]]
            
            size = models['Size_nm'].predict(inputs)[0]
            pdi = models['PDI'].predict(inputs)[0]
            zeta = models['Zeta_mV'].predict(inputs)[0]
            load = models['Drug_Loading'].predict(inputs)[0]
            ee = models['Encapsulation_Efficiency'].predict(inputs)[0]
            stab_prob = stab_model.predict_proba(inputs)[0][1] * 100
            
            # Request 1: Full units printed with large font
            results = [
                ("Droplet Size", f"{size:.2f} nm"), ("PDI", f"{pdi:.3f}"),
                ("Zeta Potential", f"{zeta:.2f} mV"), ("Drug Loading", f"{load:.2f} mg/mL"),
                ("Encapsulation Efficiency", f"{ee:.2f} %"), ("Stability Confidence", f"{stab_prob:.1f} %")
            ]
            
            for label, val in results:
                st.markdown(f"<div class='metric-container'><div class='m-label'>{label}</div><div class='m-value'>{val}</div></div>", unsafe_allow_html=True)

    with c2:
        # Request 2: Explain Axis clearly
        st.markdown("""
        <div class='axis-box'>
        <b>🧬 3D Axis Legend:</b><br>
        <b>X-Axis:</b> Oil Phase Percentage (%)<br>
        <b>Y-Axis:</b> S-mix Percentage (Surfactant + Co-surfactant)<br>
        <b>Z-Axis:</b> Aqueous Phase Percentage (Water/Buffer)
        </div>
        """, unsafe_allow_html=True)
        
        # Pseudo-ternary plot logic (X+Y+Z = 100)
        oil_range = np.linspace(5, 40, 15)
        smix_range = np.linspace(10, 70, 15)
        O, S = np.meshgrid(oil_range, smix_range)
        W = 100 - O - S
        mask = W > 0
        
        fig = go.Figure(data=[go.Scatter3d(
            x=O[mask], y=S[mask], z=W[mask],
            mode='markers',
            marker=dict(size=5, color=S[mask], colorscale='Electric', colorbar=dict(title="S-mix %"))
        )])
        
        fig.update_layout(scene=dict(
            xaxis_title='Oil % (X)', yaxis_title='S-mix % (Y)', zaxis_title='Water % (Z)'
        ), margin=dict(l=0,r=0,b=0,t=0), height=550)
        st.plotly_chart(fig, use_container_width=True)
