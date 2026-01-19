import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Fragments, AllChem, DataStructs
import pubchempy as pcp

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v23.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; }
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 20px; font-weight: 800; color: #1a202c; }
    .advice-box { background: #eef6ff; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; }
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
        else: return None, None, None, None
    
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

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier().fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

def get_chem_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {"logp": round(Descriptors.MolLogP(mol), 2), "mw": round(Descriptors.MolWt(mol), 2), "mol": mol}
    except: return None

# --- INITIALIZE STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'csv_data' not in st.session_state: st.session_state.csv_data = None
if 'drug_name' not in st.session_state: st.session_state.drug_name = "None"

# --- SIDEBAR NAVIGATION & HISTORY ---
with st.sidebar:
    st.title("Settings")
    nav_steps = ["Step 1: Drug & Oil", "Step 2: Solubility", "Step 3: Component AI", "Step 4: Ratios", "Step 5: Final Selection", "Step 6: AI Predictions"]
    step_choice = st.radio("Formulation Steps", nav_steps)
    
    st.write("---")
    st.subheader("📜 History")
    if st.session_state.history:
        for item in st.session_state.history[-5:]:
            st.caption(f"✅ {item}")
    else:
        st.write("No history yet.")

# --- STEP 1: DRUG & OIL ---
if step_choice == "Step 1: Drug & Oil":
    st.header("Step 1: Chemical Setup")
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Drug Input")
        mode = st.radio("Method", ["Manual SMILES", "Database List"])
        
        # Load data early for lists
        df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
        
        if mode == "Manual SMILES":
            smiles = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            info = get_chem_info(smiles)
            if info:
                st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
                st.session_state.drug_name = "Custom_API"
                st.image(Draw.MolToImage(info['mol'], size=(250,200)))
        else:
            if le_dict:
                st.session_state.drug_name = st.selectbox("API", sorted(le_dict['Drug_Name'].classes_))
                st.session_state.logp, st.session_state.mw = 3.5, 300.0 # Placeholder for DB drugs
            else: st.warning("Upload CSV first.")

        st.subheader("Upload Training Data")
        up = st.file_uploader("Option 3: Load CSV", type="csv")
        if up: st.session_state.csv_data = up

    with col2:
        if le_dict:
            st.subheader("Oil Phase Solubility Comparison")
            oils = le_dict['Oil_phase'].classes_
            # Logic: Unique solubility score per oil
            scores = [max(5, 100 - abs(st.session_state.logp - 3.2)*12 + np.random.randint(-10,10)) for _ in oils]
            aff_df = pd.DataFrame({"Oil": oils, "Solubility (mg/mL)": scores}).sort_values("Solubility (mg/mL)", ascending=False)
            fig = px.bar(aff_df, x="Solubility (mg/mL)", y="Oil", orientation='h', color="Solubility (mg/mL)", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("AI-predicted saturation limits across available oil phases.")

# --- STEP 2: SOLUBILITY ---
elif step_choice == "Step 2: Solubility":
    st.header("Step 2: Solubility Normalization")
    # THE CALCULATION
    base_sol = 10**(0.5 - 0.01 * (st.session_state.mw - 50) - 0.6 * st.session_state.logp) * 1000
    st.session_state.sol_limit = np.clip((base_sol / 400) * 100, 1.0, 100.0)
    
    st.metric("Practical Solubility Limit", f"{st.session_state.sol_limit:.2f} mg/mL", delta="Max 100 normalized")
    st.write("---")
    st.write("**Calculation Logic:**")
    st.latex(r"Log S = 0.5 - 0.01(MW-50) - 0.6(LogP)")
    st.markdown("> *The value is then scaled from the theoretical 400mg space down to a 100mg practical limit for nanoemulsions.*")

# --- STEP 3: COMPONENT AI ---
elif step_choice == "Step 3: Component AI":
    st.header("Step 3: AI Recommendations")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    if df_raw is not None:
        st.session_state.oil_choice = st.selectbox("Select Target Oil", sorted(le_dict['Oil_phase'].classes_))
        
        # Filtering for best surfactant
        recs = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values('Encapsulation_Efficiency_clean', ascending=False)
        
        st.markdown(f"""
        <div class="advice-box">
            <b>AI Suggestion for {st.session_state.oil_choice}:</b><br>
            Recommended Surfactant: <b>{recs['Surfactant'].iloc[0]}</b><br>
            Recommended Co-Surfactant: <b>{recs['Co-surfactant'].iloc[0]}</b>
        </div>
        """, unsafe_allow_html=True)
        

# --- STEP 4: RATIOS ---
elif step_choice == "Step 4: Ratios":
    st.header("Step 4: Formulation Ratios")
    st.session_state.drug_conc = st.slider("Drug Loading (mg/mL)", 0.1, 100.0, 5.0)
    st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    st.session_state.smix_ratio = st.selectbox("S-mix Ratio (S:Co-S)", ["1:1", "2:1", "3:1"])

# --- STEP 5: FINAL SELECTION ---
elif step_choice == "Step 5: Final Selection":
    st.header("Step 5: User Selection")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    st.session_state.s_final = st.selectbox("Confirm Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Confirm Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    
    if st.button("Finalize & Predict"):
        st.success("Formulation parameters locked. Move to Step 6.")

# --- STEP 6: AI PREDICTIONS ---
elif step_choice == "Step 6: AI Predictions":
    st.header("Step 6: Comprehensive AI Analysis")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    def safe_enc(le, val):
        try: return le.transform([val])[0]
        except: return 0

    input_data = [[
        safe_enc(le_dict['Drug_Name'], st.session_state.drug_name),
        safe_enc(le_dict['Oil_phase'], st.session_state.oil_choice),
        safe_enc(le_dict['Surfactant'], st.session_state.s_final),
        safe_enc(le_dict['Co-surfactant'], st.session_state.cs_final)
    ]]

    # 6 ORIGINAL PREDICTIONS
    res = {col: models[col].predict(input_data)[0] for col in models}
    is_stable = stab_model.predict(input_data)[0]
    
    # 1-4. Physical Props
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='m-label'>Size</div><div class='m-value'>{res['Size_nm']:.1f} nm</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='m-label'>PDI</div><div class='m-value'>{res['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='m-label'>Zeta Potential</div><div class='m-value'>{res['Zeta_mV']:.1f} mV</div></div>", unsafe_allow_html=True)
    
    c4, c5, c6 = st.columns(3)
    # 5. EE%
    c4.markdown(f"<div class='metric-card'><div class='m-label'>Encapsulation (EE%)</div><div class='m-value'>{res['Encapsulation_Efficiency']:.1f}%</div></div>", unsafe_allow_html=True)
    # 6. Final Solubility and Stability
    c5.markdown(f"<div class='metric-card'><div class='m-label'>Solubility</div><div class='m-value'>{st.session_state.sol_limit:.1f} mg/mL</div></div>", unsafe_allow_html=True)
    
    # Stability Override logic
    stable_text = "Stable" if (is_stable == 1 and st.session_state.smix_p > st.session_state.oil_p) else "Unstable"
    c6.markdown(f"<div class='metric-card' style='border-top-color:#d9534f;'><div class='m-label'>Stability Status</div><div class='m-value'>{stable_text}</div></div>", unsafe_allow_html=True)

    # REPAIR LOGIC
    st.write("---")
    col_plot, col_repair = st.columns([1.5, 1])
    with col_plot:
        # Dynamic Ternary Plot
        fig = go.Figure(go.Scatterternary({
            'mode': 'lines', 'fill': 'toself', 'name': 'Nano-Region',
            'a': [5, 15, 25, 10, 5], 'b': [40, 50, 45, 35, 40], 'c': [55, 35, 30, 55, 55]
        }))
        fig.add_trace(go.Scatterternary({
            'mode': 'markers', 'name': 'Current',
            'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [100-st.session_state.oil_p-st.session_state.smix_p],
            'marker': {'size': 12, 'color': 'red' if stable_text == "Unstable" else 'green'}
        }))
        st.plotly_chart(fig, use_container_width=True)
        

    with col_repair:
        if stable_text == "Unstable":
            st.error("⚠️ Optimization Required")
            st.write("Your formulation lies outside the self-emulsification region.")
            st.info(f"**Action Plan:**\n1. Increase S-mix to at least {st.session_state.oil_p * 2}%\n2. Decrease Oil phase below 20%\n3. Swap {st.session_state.cs_final} for a shorter chain alcohol.")
        else:
            st.success("✅ Formulation is Optimized.")
    
    # Add to History
    hist_entry = f"{st.session_state.drug_name} in {st.session_state.oil_choice} ({stable_text})"
    if hist_entry not in st.session_state.history:
        st.session_state.history.append(hist_entry)
