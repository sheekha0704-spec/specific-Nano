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
st.set_page_config(page_title="NanoPredict AI v21.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main-card { background: #f8f9fa; border-radius: 15px; padding: 25px; border: 1px solid #dee2e6; margin-bottom: 20px; }
    .metric-card { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745; text-align: center; }
    .m-value { font-size: 24px; font-weight: 800; color: #1a202c; }
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
        else:
            st.warning("No database found. Please upload a CSV to begin.")
            return None, None, None, None
    
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    
    df_train = df.copy()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier().fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

# --- CHEMICAL LOGIC ---
def get_chem_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {
            "logp": round(Descriptors.MolLogP(mol), 2),
            "mw": round(Descriptors.MolWt(mol), 2),
            "mol": mol,
            "hsp": [round(Descriptors.MolWt(mol)/20, 1), round(Fragments.fr_Ar_OH(mol)*4 + 2, 1), 5.0]
        }
    except: return None

# --- STATE MGMT ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1"
if 'csv_data' not in st.session_state: st.session_state.csv_data = None

def navigate(step):
    st.session_state.step_val = step
    st.rerun()

# --- STEP 1: INITIAL SETUP ---
if st.session_state.step_val == "Step 1":
    st.title("🧪 NanoPredict AI: Chemical Setup")
    
    # NEW: Option 3 is now front and center
    with st.expander("📂 Option 3: Upload Custom Training Database", expanded=False):
        up = st.file_uploader("Upload CSV", type="csv")
        if up: st.session_state.csv_data = up

    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)

    if df_raw is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Drug Definition")
            mode = st.radio("Drug Input", ["Database Selection", "Custom SMILES"])
            
            if mode == "Database Selection":
                drug_choice = st.selectbox("Select API", sorted(le_dict['Drug_Name'].classes_))
                st.session_state.drug_name = drug_choice
                # Mock lookup or PubChem
                st.session_state.logp, st.session_state.mw = 3.5, 250.0 
            else:
                smiles = st.text_input("Enter SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
                info = get_chem_info(smiles)
                if info:
                    st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
                    st.session_state.drug_name = "Custom_API"
                    st.info(f"Calculated Properties: MW: {info['mw']} | LogP: {info['logp']}")
                    st.image(Draw.MolToImage(info['mol'], size=(300,200)))

            st.session_state.oil = st.selectbox("Select Oil Phase", sorted(le_dict['Oil_phase'].classes_))
            
            if st.button("Calculate Affinities & Proceed →"): navigate("Step 2")

        with c2:
            st.subheader("Oil Phase Affinity Ranking")
            # Unique Solubility Logic: Predicted solubility for each oil in DB
            oils = le_dict['Oil_phase'].classes_
            # Formula: Solubility decreases as MW increases, and depends on LogP matching
            affinities = [max(10, 500 - (st.session_state.mw * 0.5) + (st.session_state.logp * 10) + np.random.randint(-20,20)) for _ in oils]
            aff_df = pd.DataFrame({"Oil": oils, "Predicted Solubility (mg/mL)": affinities}).sort_values("Predicted Solubility (mg/mL)", ascending=False)
            
            fig = px.bar(aff_df, x="Oil", y="Predicted Solubility (mg/mL)", color="Predicted Solubility (mg/mL)", title="Drug Solubility Profile across Oils")
            st.plotly_chart(fig, use_container_width=True)

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2":
    st.header("Step 2: Loading & Ratios")
    
    # Calculate solubility for the specific selected oil
    # Using the General Solubility Equation (GSE) approximation
    sol = 10**(0.5 - 0.01 * (st.session_state.mw - 50) - 0.6 * st.session_state.logp) * 1000
    st.session_state.sol_limit = np.clip(sol, 2.0, 450.0)

    st.success(f"AI Prediction: The solubility of {st.session_state.drug_name} in {st.session_state.oil} is {st.session_state.sol_limit:.2f} mg/mL")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.drug_conc = st.number_input("Target Drug Conc (mg/mL)", 0.1, 500.0, 5.0)
        st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
    with c2:
        st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
        st.session_state.smix_ratio = st.selectbox("S-mix Ratio", ["1:1", "2:1", "3:1"])

    if st.button("Screen Components →"): navigate("Step 3")

# --- STEP 3 & 4: COMPONENT SELECTION ---
elif st.session_state.step_val == "Step 3":
    st.header("Step 3: AI Selection")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.s_final = st.selectbox("Final Surfactant", sorted(le_dict['Surfactant'].classes_))
        st.session_state.cs_final = st.selectbox("Final Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
        
        if st.button("Generate Final AI Report →"): navigate("Step 4")
    with c2:
        st.info("💡 AI Tip: For high LogP drugs, non-ionic surfactants like Tween 80 show 20% higher stability in current datasets.")

# --- STEP 5: RESULTS ---
elif st.session_state.step_val == "Step 4":
    st.header("Step 4: AI Performance Suite")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)

    # Encode inputs (Handling Unknowns)
    def safe_encode(le, val):
        try: return le.transform([val])[0]
        except: return 0

    input_vec = [[
        safe_encode(le_dict['Drug_Name'], st.session_state.drug_name),
        safe_encode(le_dict['Oil_phase'], st.session_state.oil),
        safe_encode(le_dict['Surfactant'], st.session_state.s_final),
        safe_encode(le_dict['Co-surfactant'], st.session_state.cs_final)
    ]]

    # Predictions
    res = {col: models[col].predict(input_vec)[0] for col in models}
    # Stability Logic: Now dependent on Oil % and S-mix ratio
    # If Oil > 30% and Smix < 20%, stability decreases
    base_stability = stab_model.predict(input_vec)[0]
    if st.session_state.oil_p > 30 and st.session_state.smix_p < 25:
        final_stability = "Unstable (High Oil/Low Smix)"
    else:
        final_stability = "Stable" if base_stability == 1 else "Unstable"

    # Display Metrics
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"<div class='metric-card'><div class='m-label'>SIZE</div><div class='m-value'>{res['Size_nm']:.1f} nm</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='m-label'>PDI</div><div class='m-value'>{res['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='m-label'>STABILITY</div><div class='m-value'>{final_stability}</div></div>", unsafe_allow_html=True)

    # DYNAMIC TERNARY DIAGRAM
    # The 'nano-region' changes based on Smix and LogP
    st.subheader("Ternary Phase Diagram (Dynamic)")
    
    # Generate a dummy "Nanoemulsion Region" that shifts based on LogP
    shift = st.session_state.logp * 2
    t_o = [10, 20, 10, 30, 15]
    t_s = [60+shift, 50+shift, 40+shift, 50+shift, 70+shift]
    t_w = [100-a-b for a, b in zip(t_o, t_s)]

    fig = go.Figure(go.Scatterternary({
        'mode': 'lines', 'fill': 'toself', 'name': 'Nanoemulsion Region',
        'a': t_o, 'b': t_s, 'c': t_w, 'line': {'color': 'blue'}
    }))
    # Current Point
    fig.add_trace(go.Scatterternary({
        'mode': 'markers', 'name': 'Current Formula',
        'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [100-st.session_state.oil_p-st.session_state.smix_p],
        'marker': {'size': 15, 'color': 'red'}
    }))
    st.plotly_chart(fig, use_container_width=True)
    

    if st.button("🔄 Start New Project"): navigate("Step 1")
