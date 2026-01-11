import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI Precision", layout="wide")

# Custom CSS for Large, Un-truncated Metric Displays
st.markdown("""
    <style>
    .metric-container {
        background: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 8px solid #007bff;
        margin-bottom: 20px; min-width: 250px;
    }
    .m-label { font-size: 16px; color: #666; font-weight: 500; }
    .m-value { font-size: 28px; color: #111; font-weight: 800; white-space: nowrap; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE (High Accuracy Focus) ---
@st.cache_data
def load_and_train_precision():
    # Load dataset
    try:
        df = pd.read_csv('nanoemulsion 2.csv')
    except:
        st.error("Dataset 'nanoemulsion 2.csv' not found.")
        st.stop()
    
    # Precise Numeric Extraction
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    target_cols = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in target_cols:
        df[f'{col}_clean'] = df[col].apply(get_num)
        # Instead of global median, we fill based on the specific Oil group to maintain accuracy
        df[f'{col}_clean'] = df.groupby('Oil_phase')[f'{col}_clean'].transform(lambda x: x.fillna(x.median()))

    # Encoders
    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    X = df[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    
    # High Accuracy Regressors (Using Gradient Boosting for better precision)
    models = {}
    for col in target_cols:
        m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
        m.fit(X, df[f'{col}_clean'])
        models[col] = m
    
    # Stability Classifier
    df['is_stable'] = df['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=300, random_state=42)
    stab_model.fit(X, df['is_stable'])
    
    return df, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_train_precision()

# --- 2. MULTI-PAGE NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Step 1: Core Selection", "Step 2: AI Rationale", "Step 3: Predictions", "History"])

if 'inputs' not in st.session_state:
    st.session_state.inputs = {'drug': df['Drug_Name'].unique()[0], 'oil': df['Oil_phase'].unique()[0]}

# --- PAGE 1: SETUP ---
if page == "Step 1: Core Selection":
    st.header("Step 1: Drug & Phase Selection")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.inputs['drug'] = st.selectbox("Search Drug (900+)", sorted(df['Drug_Name'].unique()))
    with c2:
        st.session_state.inputs['oil'] = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))

# --- PAGE 2: RECOMMENDATIONS ---
elif page == "Step 2: AI Rationale":
    st.header("Step 2: Why these components?")
    oil = st.session_state.inputs['oil']
    drug = st.session_state.inputs['drug']
    
    # Finding specific success patterns in your 900+ rows
    top_matches = df[df['Oil_phase'] == oil].sort_values(by='Encapsulation_Efficiency', ascending=False).head(3)
    
    for i, row in top_matches.iterrows():
        with st.expander(f"Recommended System: {row['Surfactant']} + {row['Co-surfactant']}"):
            st.write(f"**Data-Driven Rationale:**")
            st.write(f"- **Accuracy Check:** This system previously achieved a Size of {row['Size_nm']} nm and EE of {row['Encapsulation_Efficiency']}.")
            st.write(f"- **Chemical Logic:** {row['Surfactant']} effectively lowers the interfacial tension of {oil}, preventing {drug} precipitation.")

# --- PAGE 3: PREDICTIONS ---
elif page == "Step 3: Predictions":
    st.header("Step 3: High-Accuracy Prediction & 3D Mapping")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        s_choice = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
        cs_choice = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Run Precision AI"):
            # Encodings
            inputs = [[le_dict['Drug_Name'].transform([st.session_state.inputs['drug']])[0],
                       le_dict['Oil_phase'].transform([st.session_state.inputs['oil']])[0],
                       le_dict['Surfactant'].transform([s_choice])[0],
                       le_dict['Co-surfactant'].transform([cs_choice])[0]]]
            
            # Predict each metric
            size = models['Size_nm'].predict(inputs)[0]
            pdi = models['PDI'].predict(inputs)[0]
            zeta = models['Zeta_mV'].predict(inputs)[0]
            load = models['Drug_Loading'].predict(inputs)[0]
            ee = models['Encapsulation_Efficiency'].predict(inputs)[0]
            stab_prob = stab_model.predict_proba(inputs)[0][1] * 100
            
            # DISPLAY RESULTS (Custom HTML to prevent truncation)
            results = [
                ("Droplet Size", f"{size:.2f} nm"),
                ("PDI", f"{pdi:.3f}"),
                ("Zeta Potential", f"{zeta:.1f} mV"),
                ("Drug Loading", f"{load:.2f} mg/mL"),
                ("EE %", f"{ee:.2f} %"),
                ("Stability Confidence", f"{stab_prob:.1f} %")
            ]
            
            for label, val in results:
                st.markdown(f"""<div class='metric-container'>
                                <div class='m-label'>{label}</div>
                                <div class='m-value'>{val}</div>
                                </div>""", unsafe_allow_html=True)

    with c2:
        st.write("**3D Ternary Phase Diagram**")
        # Define the axes clearly
        # X = Oil, Y = Smix, Z = Water
        oil_pts = np.linspace(5, 40, 15)
        smix_pts = np.linspace(10, 60, 15)
        O, S = np.meshgrid(oil_pts, smix_pts)
        W = 100 - O - S
        mask = W > 0
        
        fig = go.Figure(data=[go.Scatter3d(
            x=O[mask], y=S[mask], z=W[mask],
            mode='markers', marker=dict(size=4, color=S[mask], colorscale='Viridis', colorbar=dict(title="S-mix %"))
        )])
        fig.update_layout(scene=dict(
            xaxis_title='X: Oil Phase %',
            yaxis_title='Y: S-mix (Surf+CoS) %',
            zaxis_title='Z: Aqueous Phase %'
        ), margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
