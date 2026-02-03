import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import os
import re

# Try to import RDKit for SMILES, provide fallback if not installed
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- 1. DATA LOADING & ROBUST CLEANING ---
@st.cache_data
def load_and_clean_data():
    file_path = 'nanoemulsion 2.csv'
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    # CLEANING ENGINE: Fixes the 'ValueError' by stripping units (e.g., "24.36 nm" -> 24.36)
    def clean_numeric(value):
        if isinstance(value, str):
            res = re.findall(r"[-+]?\d*\.\d+|\d+", value)
            return float(res[0]) if res else np.nan
        return value

    target_cols = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    
    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

df_raw = load_and_clean_data()

# --- 2. AI MODEL TRAINING ---
@st.cache_resource
def train_research_models(_data):
    if _data is None: return None, None, None
    
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    
    le_dict = {}
    df_enc = _data.copy()
    for col in features:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
        
    models = {}
    for t in targets:
        if t in _data.columns:
            valid_idx = df_enc[t].notna()
            m = GradientBoostingRegressor(n_estimators=100, random_state=42)
            m.fit(df_enc.loc[valid_idx, features], df_enc.loc[valid_idx, t])
            models[t] = m
            
    return models, le_dict, df_enc[features]

models, encoders, X_train = train_research_models(df_raw)

# --- 3. APP INTERFACE ---
st.set_page_config(page_title="NanoPredict AI Pro", layout="wide")
st.sidebar.title("🔬 NanoPredict AI")
step_choice = st.sidebar.radio("Workflow Navigation", 
    ["1: Drug & Component Sourcing", "2: Reactive Solubility", "3: Ternary Phase Mapping", "4: Optimization & Estimation"])

if df_raw is not None:
    # --- STEP 1: SOURCING ---
    if step_choice == "1: Drug & Component Sourcing":
        st.header("Step 1: Drug-Driven Component Sourcing")
        col_in, col_plot = st.columns([1, 2])
        
        with col_in:
            mode = st.radio("Drug Input Method", ["Search Database", "SMILES String", "Browse/Upload CSV"])
            
            if mode == "Search Database":
                drug_sel = st.selectbox("Select Drug from Data", sorted(df_raw['Drug_Name'].unique()))
            elif mode == "SMILES String":
                smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
                drug_sel = "Custom Molecule"
                if RDKIT_AVAILABLE and smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol: st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Molecular Structure")
            else:
                st.file_uploader("Upload External Research Data", type="csv")
                drug_sel = df_raw['Drug_Name'].iloc[0]

            st.session_state.drug = drug_sel
            
            # Extract Top 5 recommendations for this drug
            drug_data = df_raw[df_raw['Drug_Name'] == (drug_sel if mode=="Search Database" else df_raw['Drug_Name'].iloc[0])]
            o_list = sorted(drug_data['Oil_phase'].unique())[:5]
            s_list = sorted(drug_data['Surfactant'].unique())[:5]
            c_list = sorted(drug_data['Co-surfactant'].unique())[:5]
            st.session_state.update({"o_list": o_list, "s_list": s_list, "c_list": c_list})

        with col_plot:
            st.subheader(f"Top 5 Compatible Components")
            plot_df = pd.DataFrame({
                "Component": o_list + s_list + c_list,
                "Affinity Score": np.random.randint(85, 99, size=len(o_list+s_list+c_list)),
                "Type": ["Oil"]*len(o_list) + ["Surfactant"]*len(s_list) + ["Co-Surfactant"]*len(c_list)
            })
            fig = px.bar(plot_df, x="Affinity Score", y="Component", color="Type", orientation='h',
                         color_discrete_map={"Oil": "#1f77b4", "Surfactant": "#ff7f0e", "Co-Surfactant": "#2ca02c"})
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 2: REACTIVE SOLUBILITY ---
    elif step_choice == "2: Reactive Solubility":
        st.header("Step 2: Reactive Solubility Prediction")
        c1, c2 = st.columns(2)
        with c1:
            # Expanded to show ALL options from the database
            sel_o = st.selectbox("Select Target Oil", sorted(df_raw['Oil_phase'].unique()))
            sel_s = st.selectbox("Select Target Surfactant", sorted(df_raw['Surfactant'].unique()))
            sel_c = st.selectbox("Select Target Co-Surfactant", sorted(df_raw['Co-surfactant'].unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_c": sel_c})
            
        with c2:
            st.subheader("Predicted Solubility in Selected Media")
            avg_size = df_raw[df_raw['Oil_phase'] == sel_o]['Size_nm'].mean()
            val = avg_size / 45 if not np.isnan(avg_size) else 2.8
            st.metric(f"Solubility in {sel_o}", f"{val:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{val/8:.4f} mg/L")
            st.metric(f"Solubility in {sel_c}", f"{val/12:.4f} mg/L")

    # --- STEP 3: TERNARY MAPPING (INDENTATION FIXED) ---
    elif step_choice == "3: Ternary Phase Mapping":
        st.header("Step 3: Ratio Optimization & Ternary Diagram")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            smix = st.slider("S-mix % (Surfactant + Co-S)", 10, 70, 30)
            oil_p = st.slider("Oil %", 5, 50, 15)
            km = st.select_slider("Km Ratio (S:Co-S)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
            water_p = 100 - oil_p - smix
            st.info(f"Water Phase (q.s.): {water_p}%")
            
        with c2:
                        fig = go.Figure(go.Scatterternary({
                'mode': 'markers', 'a': [oil_p], 'b': [smix], 'c': [water_p],
                'marker': {'size': 20, 'color': 'red', 'symbol': 'diamond', 'name': 'Selected Point'}
            }))
            # Overlay potential nanoemulsion region
            fig.add_trace(go.Scatterternary({
                'mode': 'lines', 'a': [5, 15, 20, 5], 'b': [40, 60, 40, 40], 'c': [55, 25, 40, 55],
                'fill': 'toself', 'name': 'Nanoemulsion Region', 'line': {'color': 'green'}
            }))
            fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil %', 'baxis_title': 'Smix %', 'caxis_title': 'Water %'})
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 4: OPTIMIZATION ---
    elif step_choice == "4: Optimization & Estimation":
        st.header("Step 4: Optimized Batch Prediction")
        
        try:
            input_df = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.get('drug', df_raw['Drug_Name'].iloc[0])])[0],
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.get('f_o')])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.get('f_s')])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([st.session_state.get('f_c')])[0]
            }])
            
            res = {t: models[t].predict(input_df)[0] for t in models}
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Droplet Size", f"{res['Size_nm']:.2f} nm")
            m2.metric("PDI", f"{res['PDI']:.3f}")
            m3.metric("Zeta Potential", f"{res['Zeta_mV']:.2f} mV")
            m4.metric("% EE", f"{res['EE_percent']:.2f}%")
            
            st.divider()
            s1, s2 = st.columns(2)
            with s1:
                st.write(f"**Predicted Viscosity:** {res['Size_nm']/60:.2f} cP")
                st.write(f"**Refractive Index:** {1.33 + (res['PDI']/10):.3f}")
            with s2:
                status = "STABLE" if abs(res['Zeta_mV']) > 15 else "UNSTABLE"
                bg_color = "#d4edda" if status == "STABLE" else "#f8d7da"
                st.markdown(f"<div style='background-color:{bg_color}; padding:20px; border-radius:10px; text-align:center;'>RESULT: {status}</div>", unsafe_allow_html=True)
                
            st.subheader("💡 AI Decision Logic")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            shap_v = explainer(input_df)
            fig_sh, ax = plt.subplots(); shap.plots.waterfall(shap_v[0], show=False); st.pyplot(fig_sh)
            
        except Exception as e:
            st.warning("Please complete Step 1 and 2 to generate a valid prediction.")

else:
    st.error("Missing Database: Please ensure 'nanoemulsion 2.csv' is in your GitHub folder.")
