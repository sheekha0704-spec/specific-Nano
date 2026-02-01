import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import os
import re

# --- HELPER FUNCTIONS (Moved to top to fix NameError) ---
def get_enc(le, val):
    try:
        return le.transform([val])[0]
    except:
        return 0

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v2.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; margin-bottom: 15px;}
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 22px; font-weight: 800; color: #1a202c; }
    .status-box { padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; font-weight: 800; font-size: 26px; border: 2px solid;}
    .formula-box { background: #f8f9fa; border: 1px solid #dee2e6; padding: 20px; border-radius: 10px; margin: 10px 0; font-family: 'Courier New', monospace;}
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_resource
def load_and_prep(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        csv_path = 'nanoemulsion 2.csv'
        if os.path.exists(csv_path): df = pd.read_csv(csv_path)
        else: return None, None, None, None, None

    hlb_map = {'Tween 80': 15.0, 'Tween 20': 16.7, 'Span 80': 4.3, 'Cremophor EL': 13.5, 'Labrasol': 12.0}
    df['HLB'] = df['Surfactant'].map(hlb_map).fillna(12.0)
    
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    
    le_dict = {}
    df_train = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    features = ['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'HLB']
    X = df_train[features]
    
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict, X

# Initializing Data
df_raw, models, stab_model, le_dict, X_train = load_and_prep(st.session_state.get('csv_data'))

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 NanoPredict AI")
    step_choice = st.radio("Formulation Steps", ["Step 1: Drug & Data", "Step 2: Advanced Solubility", "Step 3: Component AI & Ratios", "Step 4: Phase Ratios", "Step 5: Finalize", "Step 6: Analysis"])

# --- STEP 1: FIXING ERROR (REORDERED) ---
if step_choice == "Step 1: Drug & Data":
    st.header("Step 1: Chemical & Data Setup")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("1. Drug Input")
        mode = st.radio("Method", ["Manual SMILES", "Database List"])
        if mode == "Manual SMILES":
            smiles = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.session_state.logp, st.session_state.mw = round(Descriptors.MolLogP(mol), 2), round(Descriptors.MolWt(mol), 2)
                st.image(Draw.MolToImage(mol, size=(250,200)))
        
        st.write("---")
        st.subheader("2. Dataset Configuration")
        up = st.file_uploader("Browse/Upload Training Data (CSV)", type="csv")
        if up: st.session_state.csv_data = up

    with col2:
        if le_dict:
            st.subheader("Oil Phase Solubility Map (0-100%)")
            oils = le_dict['Oil_phase'].classes_
            scores = [np.clip(100 - abs(st.session_state.get('logp', 3.5) - (2.5 + (i*0.5)))*15, 0, 100) for i in range(len(oils))]
            aff_df = pd.DataFrame({"Oil": oils, "Solubility (%)": scores}).sort_values("Solubility (%)", ascending=False)
            st.plotly_chart(px.bar(aff_df, x="Solubility (%)", y="Oil", range_x=[0,100], orientation='h', color="Solubility (%)", color_continuous_scale='Turbo'), use_container_width=True)

# --- STEP 2: ELABORATED SOLUBILITY ---
elif step_choice == "Step 2: Advanced Solubility":
    st.header("Step 2: Thermodynamic Solubility Analysis")
    
    logp = st.session_state.get('logp', 3.0)
    mw = st.session_state.get('mw', 300.0)

    # Elaborated Solubility Equations
    sol_water = 10**(0.5 - 0.01 * (mw - 50) - 0.6 * logp) * 1000 # mg/L
    log_ko_w = logp # Octanol-Water partition
    sol_oil = sol_water * (10**log_ko_w) # Drug in Oil estimation
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💧 Aqueous Phase")
        st.metric("Drug Solubility in Water", f"{sol_water:.4f} mg/L")
        st.latex(r"S_{w} = 10^{0.5 - 0.01(MW-50) - 0.6(LogP)}")
    
    with col2:
        st.subheader("🛢️ Organic Phase")
        st.metric("Estimated Drug Solubility in Oil", f"{sol_oil/1000:.2f} mg/mL")
        st.latex(r"S_{oil} = S_{w} \times 10^{LogP}")

    st.write("---")
    st.subheader("Partitioning Insight")
    st.info(f"With a LogP of {logp}, the drug is {10**logp:.0f} times more soluble in the oil phase than water, indicating high potential for nanoemulsion encapsulation.")

# --- STEP 3: REVISED RATIOS & Km ---
elif step_choice == "Step 3: Component AI & Ratios":
    st.header("Step 3: Component AI & Smix Optimization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.km_ratio = st.select_slider("Smix Ratio (Surfactant : Co-Surfactant)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
        st.session_state.oil_choice = st.selectbox("Select Target Oil", sorted(le_dict['Oil_phase'].classes_) if le_dict else ["None"])
    
    with col2:
        st.subheader("Calculated Smix ($K_{m}$)")
        km_val = int(st.session_state.km_ratio.split(":")[0])
        st.latex(rf"K_{{m}} = \frac{{S}}{{CoS}} = {km_val}")
        st.caption("Higher ratios typically lead to smaller droplet sizes but may affect stability.")

    st.write("---")
    if df_raw is not None:
        recs = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values('Encapsulation_Efficiency_clean', ascending=False)
        if not recs.empty:
            st.success(f"AI suggests using **{recs['Surfactant'].iloc[0]}** with {st.session_state.km_ratio} ratio.")

# --- STEP 4: PHASE RATIOS ---
elif step_choice == "Step 4: Phase Ratios":
    st.header("Step 4: Global Formulation Composition")
    st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
    
    # New Parameter: S/O Ratio
    so_ratio = st.session_state.smix_p / st.session_state.oil_p
    st.metric("S-mix to Oil Ratio", f"{so_ratio:.2f}")
    st.progress(st.session_state.water_p / 100, text=f"Water Phase: {st.session_state.water_p}%")

# --- STEP 5: FINALIZE ---
elif step_choice == "Step 5: Finalize":
    st.header("Step 5: Final Selection")
    st.session_state.s_final = st.selectbox("Confirm Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Confirm Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    if st.button("Finalize for Analysis"): st.balloons()

# --- STEP 6: AI PREDICTIONS & SURFACE MAPPING ---
elif step_choice == "Step 6: Analysis":
    st.header("Step 6: AI Predictions & Surface Mapping")

    if df_raw is None:
        st.error("Missing Data: Please upload CSV in Step 1 to train the models.")
    else:
        # Prepare Input
        input_df = pd.DataFrame([{
            'Drug_Name_enc': get_enc(le_dict['Drug_Name'], st.session_state.get('drug_name', 'None')),
            'Oil_phase_enc': get_enc(le_dict['Oil_phase'], st.session_state.get('oil_choice', 'None')),
            'Surfactant_enc': get_enc(le_dict['Surfactant'], st.session_state.get('s_final', 'None')),
            'Co-surfactant_enc': get_enc(le_dict['Co-surfactant'], st.session_state.get('cs_final', 'None')),
            'HLB': 12.0
        }])

        # Stability result prominence
        is_stable = stab_model.predict(input_df)[0]
        stable_text = "STABLE NANOEMULSION" if is_stable == 1 else "POTENTIAL PHASE SEPARATION"
        
        bg_color = "#d4edda" if is_stable == 1 else "#f8d7da"
        txt_color = "#155724" if is_stable == 1 else "#721c24"
        border_color = "#28a745" if is_stable == 1 else "#dc3545"

        st.markdown(f"""
            <div class="status-box" style="background-color: {bg_color}; color: {txt_color}; border-color: {border_color};">
                {stable_text}
            </div>
            """, unsafe_allow_html=True)

        # Core Metrics
        st.subheader("📋 Predicted Physical Characterization")
        p_cols = st.columns(4)
        target_metrics = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
        
        for i, target in enumerate(target_metrics):
            prediction = models[target].predict(input_df)[0]
            with p_cols[i]:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='m-label'>{target.replace('_',' ')}</div>
                        <div class='m-value'>{prediction:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Visualizations
        tab1, tab2 = st.tabs(["📐 Ternary Phase Diagram", "🌊 3D Response Surface"])

        with tab1:
            
            fig_tern = go.Figure(go.Scatterternary({
                'mode': 'markers',
                'a': [st.session_state.get('oil_p', 15)],
                'b': [st.session_state.get('smix_p', 30)],
                'c': [st.session_state.get('water_p', 55)],
                'marker': {
                    'symbol': "circle",
                    'color': "green" if is_stable == 1 else "red",
                    'size': 18,
                    'line': {'width': 2, 'color': 'white'}
                }
            }))
            fig_tern.update_layout(ternary={'sum': 100, 'aaxis':{'title': 'Oil %'}, 'baxis':{'title': 'Smix %'}, 'caxis':{'title': 'Water %'}})
            st.plotly_chart(fig_tern, use_container_width=True)

        with tab2:
            
            st.subheader("Surface Response: Size (nm) Optimization")
            o_range = np.linspace(5, 40, 20)
            s_range = np.linspace(10, 60, 20)
            O_mesh, S_mesh = np.meshgrid(o_range, s_range)
            base_size = models['Size_nm'].predict(input_df)[0]
            Z_mesh = base_size + (O_mesh * 0.4) - (S_mesh * 0.25)
            
            fig_surf = go.Figure(data=[go.Surface(z=Z_mesh, x=o_range, y=s_range, colorscale='Viridis')])
            fig_surf.update_layout(scene=dict(xaxis_title='Oil %', yaxis_title='Smix %', zaxis_title='Predicted Size (nm)'))
            st.plotly_chart(fig_surf, use_container_width=True)

        if st.toggle("Show AI Decision Logic (SHAP Analysis)"):
            explainer = shap.Explainer(models['Size_nm'], X_train)
            shap_values = explainer(input_df)
            fig_shap, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig_shap)
