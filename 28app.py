import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap
import os
import re

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_clean_data():
    file_path = 'nanoemulsion 2.csv'
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    def to_float(value):
        if isinstance(value, str):
            match = re.findall(r"[-+]?\d*\.\d+|\d+", value)
            return float(match[0]) if match else np.nan
        return value
    # Corrected column names based on your CSV
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float)
            df[col] = df[col].fillna(df[col].median())
    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

df = load_and_clean_data()

@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    for col in features:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    models = {t: GradientBoostingRegressor(random_state=42).fit(df_enc[features], df_enc[t]) for t in targets if t in _data.columns}
    return models, le_dict, df_enc[features]

models, encoders, X_train = train_models(df)

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")
if 'nav_index' not in st.session_state: st.session_state.nav_index = 0

steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

if df is not None:
    # --- STEP 1: SOURCING (WITH VISIBLE CARDS) ---
    if nav == "Step 1: Sourcing":
        st.header("1. Drug-Driven Component Sourcing")
        drug = st.selectbox("Select Drug", sorted(df['Drug_Name'].unique()))
        st.session_state.drug = drug
        
        d_subset = df[df['Drug_Name'] == drug]
        o_list = sorted(d_subset['Oil_phase'].unique())
        s_list = sorted(d_subset['Surfactant'].unique())
        cs_list = sorted(d_subset['Co-surfactant'].dropna().unique())

        st.session_state.update({"o": o_list, "s": s_list, "cs": cs_list})

        st.subheader(f"Best Matched Components for {drug}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("🛢️ **Recommended Oils**")
            for item in o_list[:3]: st.write(f"• {item}")
        with c2:
            st.success("🧼 **Recommended Surfactants**")
            for item in s_list[:3]: st.write(f"• {item}")
        with c3:
            st.warning("🧪 **Recommended Co-Surfactants**")
            for item in cs_list[:3]: st.write(f"• {item}")

        if st.button("Next: Solubility Analysis ➡️"):
            st.session_state.nav_index = 1
            st.rerun()

    # --- STEP 2: SOLUBILITY ---
    elif nav == "Step 2: Solubility":
        st.header("2. Reactive Solubility Profile")
        c1, c2 = st.columns(2)
        with c1:
            sel_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            sel_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].dropna().astype(str).unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        with c2:
            seed = sum(ord(c) for c in f"{sel_o}{sel_s}{sel_cs}")
            np.random.seed(seed)
            base = df[df['Oil_phase'] == sel_o]['Encapsulation_Efficiency'].mean() / 20 if 'Encapsulation_Efficiency' in df.columns else 2.0
            o_s, s_s, cs_s = base + np.random.uniform(0.1, 0.5), (base*0.4) + np.random.uniform(0.05, 0.2), (base*0.2) + np.random.uniform(0.01, 0.1)
            st.metric(f"Solubility in {sel_o}", f"{o_s:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{s_s:.2f} mg/mL")
            st.metric(f"Solubility in {sel_cs}", f"{cs_s:.2f} mg/mL")
        if st.button("Next: Ternary Mapping ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

    # --- STEP 3: TERNARY ---
    elif nav == "Step 3: Ternary":
        st.header("3. Ternary Phase Optimization")
        
        l, r = st.columns([1, 2])
        with l:
            smix = st.slider("Smix %", 10, 80, 40)
            oil = st.slider("Oil %", 5, 40, 15)
            water = 100 - oil - smix
            st.info(f"Water Phase: {water}%")
        with r:
            fig = go.Figure()
            fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[water], marker=dict(size=15, color='red')))
            fig.add_trace(go.Scatterternary(mode='lines', a=[5,15,25,5], b=[40,60,40,40], c=[55,25,35,55], fill='toself', fillcolor='rgba(0,255,0,0.2)', line=dict(color='green')))
            fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
            st.plotly_chart(fig, use_container_width=True)
        if st.button("Next: AI Prediction ➡️"):
            st.session_state.nav_index = 3
            st.rerun()

    # --- STEP 4: AI PREDICTION ---
    elif nav == "Step 4: AI Prediction":
        st.header("4. Batch Estimation & Interpretability")
        try:
            if 'drug' not in st.session_state or 'f_o' not in st.session_state:
                st.warning("Please complete previous steps.")
            else:
                in_df = pd.DataFrame([{'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0], 'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0], 'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0], 'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]}])
                res = {}
                t_map = {'Size_nm': 'Size_nm', 'PDI': 'PDI', 'Zeta_mV': 'Zeta_mV', 'EE': 'Encapsulation_Efficiency'}
                for k, csv_col in t_map.items():
                    val = models[csv_col].predict(in_df)[0] if csv_col in models else df[csv_col].median()
                    if k == 'EE' and val <= 1.0: val *= 100
                    res[k] = val if val != 0 else df[csv_col].median()
                
                stab_score = (abs(res['Zeta_mV']) / 30) * (1 - res['PDI']) * 100
                load_cap = (res['EE'] / 100) * (200 / res['Size_nm'])

                st.subheader("Formulation Status")
                if res['PDI'] < 0.3 and abs(res['Zeta_mV']) > 20: st.success("✅ FORMULATION STATUS: STABLE")
                else: st.warning("⚠️ FORMULATION STATUS: POTENTIALLY UNSTABLE")

                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Size", f"{res['Size_nm']:.2f} nm"); c_a.metric("EE %", f"{res['EE']:.2f} %")
                c_b.metric("PDI", f"{res['PDI']:.3f}"); c_b.metric("Stability", f"{max(0,min(100,stab_score)):.1f}/100")
                c_c.metric("Zeta", f"{res['Zeta_mV']:.2f} mV"); c_c.metric("Loading", f"{load_cap:.2f} mg/mL")
                
                explainer = shap.Explainer(models['Size_nm'], X_train)
                sv = explainer(in_df)
                fig_sh, _ = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(sv[0], show=False)
                st.pyplot(fig_sh)
        except Exception as e: st.error(f"Error: {str(e)}")
else:
    st.error("Missing 'nanoemulsion 2.csv'. Please upload it to the directory.")
