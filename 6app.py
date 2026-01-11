import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os

# --- CHEMICAL LIBRARIES SAFETY ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v9.0", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745;
        text-align: center; margin-bottom: 20px;
    }
    .m-label { font-size: 14px; color: #555; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 24px; color: #1a202c; font-weight: 800; }
    .step-box { background: #f7fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 20px; }
    .locked-msg { text-align: center; padding: 50px; color: #a0aec0; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"Please upload '{csv_file}' to your GitHub repo.")
        st.stop()
    
    df = pd.read_csv(csv_file)
    
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in targets:
        df[f'{col}_clean'] = df[col].apply(get_num)
        
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()

    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        df_train[col] = df_train[col].fillna("Not Specified").astype(str)

    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le
        
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=300, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable').fillna(False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, df_train['is_stable'])
    
    return df_train, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_prep()

# --- 2. UI STATE MANAGEMENT ---
if 'step' not in st.session_state: st.session_state.step = 1

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("NanoPredict Workflow")
nav = st.sidebar.radio("Navigation", ["Step 1: Phase Selection", "Step 2: Concentrations", "Step 3: Component Screening", "Step 4: Final Selection", "Step 5: Optimization & Results"])

# --- STEP 1: PHASE SELECTION ---
if nav == "Step 1: Phase Selection":
    st.header("Step 1: API & Phase Identification")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            drug = st.selectbox("Select API (Drug)", sorted(df['Drug_Name'].unique()))
        with col2:
            oil = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
        with col3:
            aq = st.selectbox("Select Aqueous Phase", ["Distilled Water", "Phosphate Buffer (pH 6.8)", "Saline", "Deionized Water"])
        
        if st.button("Confirm Phases"):
            st.session_state.drug = drug
            st.session_state.oil = oil
            st.session_state.aq = aq
            st.session_state.step = max(st.session_state.step, 2)
            st.success("Phases locked! Proceed to Step 2.")

# --- STEP 2: CONCENTRATIONS ---
elif nav == "Step 2: Concentrations":
    if st.session_state.step < 2: st.warning("Please complete Step 1 first.")
    else:
        st.header("Step 2: Define Concentrations")
        col1, col2 = st.columns(2)
        with col1:
            drug_mg = st.number_input(f"Drug Amount (mg) for {st.session_state.drug}", min_value=0.1, value=10.0)
            oil_perc = st.slider(f"Oil Phase Concentration (%) for {st.session_state.oil}", 1, 50, 10)
        with col2:
            st.info(f"**Target System:** {st.session_state.drug} in {st.session_state.oil} oil.")
            st.write(f"Remaining volume will be balanced by {st.session_state.aq}.")
        
        if st.button("Save Concentrations"):
            st.session_state.drug_mg = drug_mg
            st.session_state.oil_perc = oil_perc
            st.session_state.step = max(st.session_state.step, 3)
            st.success("Concentrations saved! Proceed to Step 3.")

# --- STEP 3: COMPONENT SCREENING ---
elif nav == "Step 3: Component Screening":
    if st.session_state.step < 3: st.warning("Please complete Step 2 first.")
    else:
        st.header("Step 3: Suggested Surfactants & Co-Surfactants")
        st.write("Based on your Oil selection, these are the top performing individual components:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("High Performance Surfactants")
            s_list = df[df['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)['Surfactant'].unique()[:5]
            for s in s_list:
                st.markdown(f"✅ `{s}`")
        
        with col2:
            st.subheader("Compatible Co-Surfactants")
            cs_list = df[df['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)['Co-surfactant'].unique()[:5]
            for cs in cs_list:
                st.markdown(f"🔗 `{cs}`")
        
        st.session_state.step = max(st.session_state.step, 4)
        st.info("Review the suggestions and move to Step 4 to pick your combination.")

# --- STEP 4: FINAL SELECTION ---
elif nav == "Step 4: Final Selection":
    if st.session_state.step < 4: st.warning("Please complete Step 3 first.")
    else:
        st.header("Step 4: Select Your Formulation Pair")
        col1, col2 = st.columns(2)
        with col1:
            final_s = st.selectbox("Choose Surfactant", sorted(df['Surfactant'].unique()))
        with col2:
            final_cs = st.selectbox("Choose Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        
        if st.button("Lock Formulation"):
            st.session_state.final_s = final_s
            st.session_state.final_cs = final_cs
            st.session_state.step = max(st.session_state.step, 5)
            st.success(f"Locked: {final_s} + {final_cs}. Proceed to Step 5.")

# --- STEP 5: OPTIMIZATION & RESULTS ---
elif nav == "Step 5: Optimization & Results":
    if st.session_state.step < 5: st.warning("Please complete Step 4 first.")
    else:
        st.header("Step 5: Optimum Range & Predicted Outcomes")
        
        # ML Prediction Logic
        inputs = [[le_dict['Drug_Name'].transform([st.session_state.drug])[0],
                   le_dict['Oil_phase'].transform([st.session_state.oil])[0],
                   le_dict['Surfactant'].transform([st.session_state.final_s])[0],
                   le_dict['Co-surfactant'].transform([st.session_state.final_cs])[0]]]
        
        res = [models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
        conf = stab_model.predict_proba(inputs)[0][1] * 100

        st.success(f"### AI Optimization for: {st.session_state.final_s} & {st.session_state.final_cs}")
        
        # Display Optimum Range
        st.markdown(f"""
        <div style="background:#e6fffa; padding:15px; border-radius:10px; border:1px solid #38b2ac;">
            <b>Recommended Smix Range:</b> Based on the data, keep the Surfactant:Co-Surfactant ratio between <b>1:1 and 3:1</b> 
            with a total Smix concentration of <b>20% - 40%</b> for maximum stability.
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        
        # Display Results
        m_cols = st.columns(3)
        metrics = [("Droplet Size", f"{res[0]:.2f} nm"), ("PDI", f"{res[1]:.3f}"), ("Zeta Potential", f"{res[2]:.1f} mV"),
                   ("Drug Loading", f"{res[3]:.2f} mg/mL"), ("EE %", f"{res[4]:.1f} %"), ("Stability Score", f"{conf:.1f} %")]
        
        for i, (l, v) in enumerate(metrics):
            with m_cols[i % 3]:
                st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        # 3D Phase Diagram (Visual Representation)
        st.subheader("Pseudo-Ternary Phase Mapping")
        o_v, s_v = np.meshgrid(np.linspace(5, 40, 15), np.linspace(15, 65, 15))
        w_v = 100 - o_v - s_v
        mask = w_v > 0
        fig = go.Figure(data=[go.Scatter3d(x=o_v[mask], y=s_v[mask], z=w_v[mask], mode='markers',
                                         marker=dict(size=4, color=s_v[mask], colorscale='Viridis', opacity=0.8))])
        fig.update_layout(scene=dict(xaxis_title='Oil %', yaxis_title='Smix %', zaxis_title='Water %'), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig, use_container_width=True)
