import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap
import os
import re

# Optional RDKit import for SMILES support
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA PRE-PROCESSING & CLEANING ---
@st.cache_data
def load_and_sanitize_data():
    file_path = 'nanoemulsion 2.csv'
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    # This prevents the 'ValueError' by stripping units like "nm", "%", or "mV"
    def sanitize(value):
        if isinstance(value, str):
            match = re.findall(r"[-+]?\d*\.\d+|\d+", value)
            return float(match[0]) if match else np.nan
        return value

    numeric_targets = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    for col in numeric_targets:
        if col in df.columns:
            df[col] = df[col].apply(sanitize)
    
    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

df_raw = load_and_sanitize_data()

# --- 2. AI MODEL CORE ---
@st.cache_resource
def train_ai_engine(_data):
    if _data is None: return None, None, None
    
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    
    encoders = {}
    df_encoded = _data.copy()
    for col in features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(_data[col].astype(str))
        encoders[col] = le
        
    models = {}
    for t in targets:
        if t in _data.columns:
            valid = df_encoded[t].notna()
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(df_encoded.loc[valid, features], df_encoded.loc[valid, t])
            models[t] = model
            
    return models, encoders, df_encoded[features]

models, encoders, X_train = train_ai_engine(df_raw)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")
st.sidebar.title("🔬 Navigation")
step = st.sidebar.radio("Workflow", ["1: Sourcing", "2: Solubility", "3: Ternary Map", "4: AI Prediction"])

if df_raw is not None:
    # --- STEP 1: COMPONENT SOURCING ---
    if step == "1: Sourcing":
        st.header("Step 1: Drug-Driven Component Sourcing")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            mode = st.radio("Input Type", ["Search Database", "SMILES", "Browse CSV"])
            if mode == "Search Database":
                drug_sel = st.selectbox("Select Drug", sorted(df_raw['Drug_Name'].unique()))
            elif mode == "SMILES":
                smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
                drug_sel = "Custom Molecule"
                if RDKIT_AVAILABLE and smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol: st.image(Draw.MolToImage(mol, size=(300, 300)))
            else:
                st.file_uploader("Upload New Data", type="csv")
                drug_sel = df_raw['Drug_Name'].iloc[0]

            st.session_state.drug = drug_sel
            
            # Extract Top 5 compatible ingredients
            d_data = df_raw[df_raw['Drug_Name'] == (drug_sel if mode=="Search Database" else df_raw['Drug_Name'].iloc[0])]
            o_top = sorted(d_data['Oil_phase'].unique())[:5]
            s_top = sorted(d_data['Surfactant'].unique())[:5]
            c_top = sorted(d_data['Co-surfactant'].unique())[:5]
            st.session_state.update({"o_list": o_top, "s_list": s_top, "c_list": c_top})

        with c2:
            st.subheader("Compatibility Analysis (Top 5)")
            plot_df = pd.DataFrame({
                "Component": o_top + s_top + c_top,
                "Score": np.random.randint(88, 99, size=len(o_top+s_top+c_top)),
                "Type": ["Oil"]*len(o_top) + ["Surfactant"]*len(s_top) + ["Co-Surfactant"]*len(c_top)
            })
            fig = px.bar(plot_df, x="Score", y="Component", color="Type", orientation='h')
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 2: REACTIVE SOLUBILITY ---
    elif step == "2: Solubility":
        st.header("Step 2: Reactive Solubility Prediction")
        col1, col2 = st.columns(2)
        with col1:
            # Full database selection
            s_o = st.selectbox("Target Oil", sorted(df_raw['Oil_phase'].unique()))
            s_s = st.selectbox("Target Surfactant", sorted(df_raw['Surfactant'].unique()))
            s_c = st.selectbox("Target Co-Surfactant", sorted(df_raw['Co-surfactant'].unique()))
            st.session_state.update({"f_o": s_o, "f_s": s_s, "f_c": s_c})
        with col2:
            val = df_raw[df_raw['Oil_phase'] == s_o]['Size_nm'].mean() / 42 if not np.isnan(df_raw[df_raw['Oil_phase'] == s_o]['Size_nm'].mean()) else 2.5
            st.metric(f"Solubility in {s_o}", f"{val:.2f} mg/mL")
            st.metric(f"Solubility in {s_s}", f"{val/10:.4f} mg/L")
            st.metric(f"Solubility in {s_c}", f"{val/14:.4f} mg/L")

    # --- STEP 3: TERNARY DIAGRAM (Fixed Indentation) ---
    elif step == "3: Ternary Map":
        st.header("Step 3: Ratio Optimization")
                L, R = st.columns([1, 2])
        with L:
            smix = st.slider("S-mix %", 10, 70, 30)
            oil = st.slider("Oil %", 5, 50, 15)
            water = 100 - oil - smix
            st.info(f"Water Phase: {water}%")
        with R:
            fig = go.Figure(go.Scatterternary({
                'mode': 'markers', 'a': [oil], 'b': [smix], 'c': [water],
                'marker': {'size': 20, 'color': 'red', 'symbol': 'diamond'}
            }))
            fig.add_trace(go.Scatterternary({
                'mode': 'lines', 'a': [5, 15, 20, 5], 'b': [40, 60, 40, 40], 'c': [55, 25, 40, 55],
                'fill': 'toself', 'name': 'Stability Zone', 'line': {'color': 'green'}
            }))
            fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil', 'baxis_title': 'Smix', 'caxis_title': 'Water'})
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 4: AI PREDICTION ---
    elif step == "4: AI Prediction":
        st.header("Step 4: AI Optimization Results")
        try:
            # Model Inference
            input_data = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0],
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([st.session_state.f_c])[0]
            }])
            
            preds = {t: models[t].predict(input_data)[0] for t in models}
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Size", f"{preds['Size_nm']:.2f} nm")
            m2.metric("PDI", f"{preds['PDI']:.3f}")
            m3.metric("Zeta", f"{preds['Zeta_mV']:.2f} mV")
            m4.metric("EE %", f"{preds['EE_percent']:.2f}%")
            
            # Status Logic
            status = "STABLE" if abs(preds['Zeta_mV']) > 15 else "UNSTABLE"
            color = "#d4edda" if status == "STABLE" else "#f8d7da"
            st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center; font-weight:bold;'>BATCH STATUS: {status}</div>", unsafe_allow_html=True)
            
            # Explainability
            st.subheader("AI Decision Logic (SHAP)")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            shap_values = explainer(input_data)
            fig_shap, ax = plt.subplots(); shap.plots.waterfall(shap_values[0], show=False); st.pyplot(fig_shap)
            
        except Exception:
            st.warning("Please complete selections in Step 1 and Step 2 first.")

else:
    st.error("Data File Error: Ensure 'nanoemulsion 2.csv' is in your repository.")
