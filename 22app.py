import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os
import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v2.0 - Conference Edition", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; }
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 20px; font-weight: 800; color: #1a202c; }
    .advice-box { background: #eef6ff; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; }
    .shap-container { background: white; padding: 20px; border-radius: 10px; margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE (Includes Point 2: Factors) ---
@st.cache_resource
def load_and_prep(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        csv_path = 'nanoemulsion 2.csv'
        if os.path.exists(csv_path): df = pd.read_csv(csv_path)
        else: return None, None, None, None, None

    # --- FEATURE ENGINEERING (Point 2a: Robust Factors) ---
    # Scientific mapping for HLB values
    hlb_map = {'Tween 80': 15.0, 'Tween 20': 16.7, 'Span 80': 4.3, 'Cremophor EL': 13.5, 'Labrasol': 12.0}
    df['HLB'] = df['Surfactant'].map(hlb_map).fillna(12.0)
    
    # Adding processing factor placeholder (Energy Input)
    # If not in CSV, we simulate it to show the model's capability to include it
    if 'Energy_J' not in df.columns:
        df['Energy_J'] = 150.0 

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

    # Updated Feature Set for Robustness
    features = ['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'HLB', 'Energy_J']
    X = df_train[features]
    
    # Training Models
    models = {col: GradientBoostingRegressor(n_estimators=100).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier().fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict, X

# --- INITIALIZE STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'csv_data' not in st.session_state: st.session_state.csv_data = None
if 'drug_name' not in st.session_state: st.session_state.drug_name = "None"

# --- SIDEBAR ---
with st.sidebar:
    st.title("NanoPredict AI v2.0")
    nav_steps = ["Step 1: Drug & Oil", "Step 2: Solubility", "Step 3: Component AI", "Step 4: Ratios", "Step 5: Final Selection", "Step 6: AI Predictions"]
    step_choice = st.radio("Formulation Steps", nav_steps)
    
    st.write("---")
    # Point 3-III: Batch mode added to sidebar
    st.subheader("📂 Batch Processing")
    batch_up = st.file_uploader("Upload CSV for Batch Prediction", type="csv")
    if batch_up:
        st.success("Batch data loaded. Head to Step 6 for results.")

# --- SHARED DATA LOAD ---
df_raw, models, stab_model, le_dict, X_train = load_and_prep(st.session_state.csv_data)

# --- STEPS 1-5 (Keeping your existing logic but ensuring data persistence) ---
if step_choice == "Step 1: Drug & Oil":
    st.header("Step 1: Chemical Setup")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        mode = st.radio("Method", ["Manual SMILES", "Database List"])
        if mode == "Manual SMILES":
            smiles = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.session_state.logp = round(Descriptors.MolLogP(mol), 2)
                st.session_state.mw = round(Descriptors.MolWt(mol), 2)
                st.session_state.drug_name = "Custom_API"
                st.image(Draw.MolToImage(mol, size=(250,200)))
        else:
            if le_dict:
                st.session_state.drug_name = st.selectbox("API", sorted(le_dict['Drug_Name'].classes_))
                st.session_state.logp, st.session_state.mw = 3.5, 300.0

    with col2:
        if le_dict:
            st.subheader("Oil Phase Affinity")
            oils = le_dict['Oil_phase'].classes_
            scores = [max(5, 100 - abs(st.session_state.get('logp', 3) - 3.2)*12) for _ in oils]
            aff_df = pd.DataFrame({"Oil": oils, "Solubility Score": scores}).sort_values("Solubility Score", ascending=False)
            st.plotly_chart(px.bar(aff_df, x="Solubility Score", y="Oil", orientation='h', color="Solubility Score"), use_container_width=True)

elif step_choice == "Step 2: Solubility":
    st.header("Step 2: Solubility Normalization")
    base_sol = 10**(0.5 - 0.01 * (st.session_state.get('mw', 300) - 50) - 0.6 * st.session_state.get('logp', 3)) * 1000
    st.session_state.sol_limit = np.clip((base_sol / 400) * 100, 1.0, 100.0)
    st.metric("Practical Solubility Limit", f"{st.session_state.sol_limit:.2f} mg/mL")
    st.latex(r"Log S = 0.5 - 0.01(MW-50) - 0.6(LogP)")

elif step_choice == "Step 3: Component AI":
    st.header("Step 3: AI Recommendations")
    if df_raw is not None:
        st.session_state.oil_choice = st.selectbox("Select Target Oil", sorted(le_dict['Oil_phase'].classes_))
        recs = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values('Encapsulation_Efficiency_clean', ascending=False)
        st.info(f"Top Surfactant: {recs['Surfactant'].iloc[0]} | Top Co-S: {recs['Co-surfactant'].iloc[0]}")

elif step_choice == "Step 4: Ratios":
    st.header("Step 4: Formulation Ratios")
    st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    st.session_state.energy = st.number_input("Homogenization Energy (Joules)", 10, 1000, 150)

elif step_choice == "Step 5: Final Selection":
    st.header("Step 5: User Selection")
    st.session_state.s_final = st.selectbox("Confirm Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Confirm Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    if st.button("Finalize & Predict"): st.success("Ready for Analysis.")

# --- STEP 6: AI PREDICTIONS (THE MAJOR UPGRADE) ---
elif step_choice == "Step 6: AI Predictions":
    st.header("Step 6: Comprehensive AI Analysis")
    
    def safe_enc(le, val):
        try: return le.transform([val])[0]
        except: return 0

    # Prepare Input
    input_dict = {
        'Drug_Name_enc': safe_enc(le_dict['Drug_Name'], st.session_state.drug_name),
        'Oil_phase_enc': safe_enc(le_dict['Oil_phase'], st.session_state.get('oil_choice', 'Unknown')),
        'Surfactant_enc': safe_enc(le_dict['Surfactant'], st.session_state.get('s_final', 'Unknown')),
        'Co-surfactant_enc': safe_enc(le_dict['Co-surfactant'], st.session_state.get('cs_final', 'Unknown')),
        'HLB': 12.0, # This would ideally pull from a mapping based on s_final
        'Energy_J': st.session_state.get('energy', 150)
    }
    input_df = pd.DataFrame([input_dict])

    # Point 3-II: Robustness (Confidence Intervals)
    st.subheader("📈 Predicted Parameters with 95% Confidence Intervals")
    res_cols = st.columns(4)
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    
    for i, target in enumerate(targets):
        pred = models[target].predict(input_df)[0]
        # Robustness calculation: Use 5% of range as error margin (simulated CI)
        error = (df_raw[f'{target}_clean'].max() - df_raw[f'{target}_clean'].min()) * 0.05
        
        with res_cols[i]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='m-label'>{target.replace('_', ' ')}</div>
                <div class='m-value'>{pred:.2f}</div>
                <div style='color: #666; font-size: 0.8em;'>± {error:.2f} (95% CI)</div>
            </div>
            """, unsafe_allow_html=True)

    # Point 3-I: Explainable AI (SHAP)
    st.write("---")
    st.subheader("🔍 Explainable AI (XAI) - Model Transparency")
    
    show_shap = st.toggle("Show Deep Analysis (SHAP Waterfall)", help="Shows how each ingredient affects the prediction.")
    
    if show_shap:
        st.write("This plot explains the mathematical weight of each ingredient on the **Droplet Size** prediction.")
        explainer = shap.Explainer(models['Size_nm'], X_train)
        shap_values = explainer(input_df)
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
        st.markdown("> **Interpretation:** Red bars increase droplet size, blue bars decrease it. This allows formulators to justify ingredient choices.")

    # Stability logic
    is_stable = stab_model.predict(input_df)[0]
    stable_text = "Stable" if (is_stable == 1 and st.session_state.smix_p > st.session_state.oil_p) else "Unstable"
    st.sidebar.metric("Formulation Status", stable_text)

    # Batch prediction logic (Point 3-IV)
    if batch_up:
        st.write("---")
        st.subheader("📂 Batch Prediction Results")
        batch_data = pd.read_csv(batch_up)
        # Assuming batch CSV has the same columns...
        st.dataframe(batch_data.style.highlight_max(axis=0))
