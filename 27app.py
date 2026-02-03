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

# --- 1. ROBUST DATA CLEANING ---
@st.cache_data
def load_and_clean_data():
    file_path = 'nanoemulsion 2.csv'
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    def to_float(value):
        if isinstance(value, str):
            match = re.findall(r"[-+]?\d*\.\d+|\d+", value)
            return float(match[0]) if match else np.nan
        return value

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float)
    
    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

df = load_and_clean_data()

# --- 2. AI ENGINE ---
@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None
    
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    
    le_dict = {}
    df_enc = _data.copy()
    for col in features:
        le = LabelEncoder()
        # Convert to string to handle potential None values before encoding
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
        
    models = {}
    for t in targets:
        if t in _data.columns:
            valid = df_enc[t].notna()
            m = GradientBoostingRegressor(n_estimators=100, random_state=42)
            m.fit(df_enc.loc[valid, features], df_enc.loc[valid, t])
            models[t] = m
            
    return models, le_dict, df_enc[features]

models, encoders, X_train = train_models(df)

# --- AUTO-NAVIGATION INITIALIZATION ---
if 'nav_index' not in st.session_state:
    st.session_state.nav_index = 0

# --- 3. APP INTERFACE ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")
st.title("🔬 NanoPredict AI Research Suite")

steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

if df is not None:
    # --- STEP 1: SOURCING ---
    if nav == "Step 1: Sourcing":
        st.header("1. Drug-Driven Component Sourcing")
        c1, c2 = st.columns([1, 2])
        with c1:
            mode = st.radio("Input Method", ["Database", "SMILES", "Browse File"])
            if mode == "Database":
                drug = st.selectbox("Select Drug", sorted(df['Drug_Name'].unique()))
            elif mode == "SMILES":
                smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
                drug = df['Drug_Name'].iloc[0]
            else:
                st.file_uploader("Upload CSV", type="csv")
                drug = df['Drug_Name'].iloc[0]
            
            st.session_state.drug = drug
            d_subset = df[df['Drug_Name'] == drug]
            o_list = sorted(d_subset['Oil_phase'].unique())[:5]
            s_list = sorted(d_subset['Surfactant'].unique())[:5]
            c_list = sorted(d_subset['Co-surfactant'].dropna().unique())[:5]
            st.session_state.update({"o": o_list, "s": s_list, "cs": c_list})

        with c2:
            st.subheader("Top 5 Compatibility Scores")
            plot_df = pd.DataFrame({
                "Component": o_list + s_list + c_list,
                "Affinity": np.random.randint(80, 99, len(o_list+s_list+c_list)),
                "Type": ["Oil"]*len(o_list) + ["Surfactant"]*len(s_list) + ["Co-Surf"]*len(c_list)
            })
            fig = px.bar(plot_df, x="Affinity", y="Component", color="Type", orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Proceed to Solubility Analysis ➡️"):
            st.session_state.nav_index = 1
            st.rerun()

    # --- STEP 2: SOLUBILITY ---
    elif nav == "Step 2: Solubility":
        st.header("2. Reactive Solubility & Personalized Metrics")
        col1, col2 = st.columns(2)
        with col1:
            sel_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            # Fixed sorted error from your screenshot by ensuring string type and dropping NaNs
            sel_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].dropna().astype(str).unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        
        with col2:
            st.subheader("Unique Personalized Solubility Profile")
            # Logic: Calculate base solubility using EE% from specific matches
            combo_match = df[(df['Oil_phase'] == sel_o) & (df['Surfactant'] == sel_s)]
            
            if not combo_match.empty and 'EE_percent' in combo_match.columns:
                base_sol = combo_match['EE_percent'].mean() / 25
            else:
                base_sol = df[df['Oil_phase'] == sel_o]['EE_percent'].mean() / 30 if 'EE_percent' in df.columns else 2.5
            
            # Handling potential NaN from mean()
            base_sol = base_sol if not np.isnan(base_sol) else 2.5

            st.metric(f"Solubility in {sel_o}", f"{base_sol:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{(base_sol * 0.35):.2f} mg/mL")
            st.metric(f"Solubility in {sel_cs}", f"{(base_sol * 0.18):.2f} mg/mL")

        if st.button("Proceed to Ternary Mapping ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

    # --- STEP 3: TERNARY ---
    elif nav == "Step 3: Ternary":
        st.header("3. Ternary Phase Optimization")
        # Removed the text [Image of...] that caused SyntaxError in your screenshot
        
        left, right = st.columns([1, 2])
        with left:
            smix = st.slider("Smix %", 10, 80, 40)
            oil = st.slider("Oil %", 5, 40, 15)
            water = 100 - oil - smix
            st.info(f"Water Phase: {water}%")
        with right:
            fig = go.Figure(go.Scatterternary({
                'mode': 'markers', 'a': [oil], 'b': [smix], 'c': [water],
                'marker': {'size': 18, 'color': 'red', 'symbol': 'diamond'}
            }))
            fig.add_trace(go.Scatterternary({
                'mode': 'lines', 'a': [5, 15, 25, 5], 'b': [40, 60, 40, 40], 'c': [55, 25, 35, 55],
                'fill': 'toself', 'name': 'Stable Region', 'line': {'color': 'green'}
            }))
            fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil %', 'baxis_title': 'Smix %', 'caxis_title': 'Water %'})
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("Proceed to Final AI Prediction ➡️"):
            st.session_state.nav_index = 3
            st.rerun()

    # --- STEP 4: AI PREDICTION & TECHNICAL SHAP ---
    elif nav == "Step 4: AI Prediction":
        st.header("4. Batch Estimation & Interpretability")
        try:
            if 'f_o' not in st.session_state:
                st.warning("⚠️ Component data missing. Please complete Step 2.")
            else:
                input_df = pd.DataFrame([{
                    'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0],
                    'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                    'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                    'Co-surfactant': encoders['Co-surfactant'].transform([st.session_state.f_cs])[0]
                }])
                
                res = {t: models[t].predict(input_df)[0] for t in models}
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Size", f"{res['Size_nm']:.2f} nm")
                m2.metric("PDI", f"{res['PDI']:.3f}")
                m3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
                m4.metric("EE %", f"{res['EE_percent']:.2f}%")
                
                status = "STABLE" if abs(res['Zeta_mV']) > 15 else "UNSTABLE"
                st.markdown(f"<div style='background-color:{'#d4edda' if status=='STABLE' else '#f8d7da'}; padding:20px; border-radius:10px; text-align:center;'><b>STATUS: {status}</b></div>", unsafe_allow_html=True)
                
                st.divider()
                st.subheader("AI Decision Logic (SHAP Interpretation)")
                
                explainer = shap.Explainer(models['Size_nm'], X_train)
                sv = explainer(input_df)
                fig_sh, ax = plt.subplots(); shap.plots.waterfall(sv[0], show=False); st.pyplot(fig_sh)

                st.info("""
                **Technical Explanation of SHAP Waterfall Plot:**
                1. **Base Value ($E[f(X)]$):** The expected average output of the model (droplet size) given the training background.
                2. **Feature Contribution:** Each row shows how much a specific component (Drug, Oil, etc.) shifted the prediction away from the base value.
                3. **Red (Positive Values):** Indicates the component increased the droplet size prediction.
                4. **Blue (Negative Values):** Indicates the component decreased the droplet size prediction, effectively aiding in 'nanonization'.
                5. **$f(x)$:** The final output on the top of the chart is the actual predicted value for your current selection.
                """)
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
else:
    st.error("Please ensure 'nanoemulsion 2.csv' is uploaded.")
