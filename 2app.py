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
st.set_page_config(page_title="NanoPredict AI v8.0", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background: #ffffff; padding: 22px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 10px solid #28a745;
        margin-bottom: 20px;
    }
    .m-label { font-size: 14px; color: #555; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 26px; color: #000; font-weight: 800; white-space: nowrap; }
    .outlier-box { background: #fff5f5; border: 1px solid #feb2b2; padding: 15px; border-radius: 10px; margin-top: 10px; }
    .locked-msg { text-align: center; padding: 50px; color: #a0aec0; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE & OUTLIER LOGIC ---
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
    
    # IDENTIFY OUTLIERS (Statistical Anomaly Detection)
    # Finding rows where EE% or Size is outside the 1.5x IQR range
    q1 = df_train['Encapsulation_Efficiency_clean'].quantile(0.25)
    q3 = df_train['Encapsulation_Efficiency_clean'].quantile(0.75)
    iqr = q3 - q1
    df_train['is_outlier'] = (df_train['Encapsulation_Efficiency_clean'] < (q1 - 1.5 * iqr)) | \
                             (df_train['Encapsulation_Efficiency_clean'] > (q3 + 1.5 * iqr))

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

# --- 2. STRUCTURE PREDICTION ---
@st.cache_data
def get_structure(drug_name):
    if not HAS_CHEM_LIBS: return None, None, None, None
    try:
        comp = pcp.get_compounds(drug_name, 'name')[0]
        mol = Chem.MolFromSmiles(comp.canonical_smiles)
        return Draw.MolToImage(mol, size=(300, 300)), comp.canonical_smiles, comp.molecular_weight, comp.xlogp
    except: return None, None, None, None

# --- 3. UI STATE MANAGEMENT ---
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False

# SIDEBAR
st.sidebar.title("Navigation Controls")
nav_choice = st.sidebar.radio("Go to:", ["Step 1: Chemical Setup", "Step 2: Expert Rationale", "Step 3: Outcome Prediction"])

# --- PAGE 1: SETUP (Always Active) ---
if nav_choice == "Step 1: Chemical Setup":
    st.header("Step 1: API & Structural Analysis")
    c1, c2 = st.columns(2)
    with c1:
        selected_drug = st.selectbox("Select API", sorted(df['Drug_Name'].unique()), key="drug_select")
        selected_oil = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()), key="oil_select")
        
        if st.button("✅ Confirm Selections & Unlock Next Steps"):
            st.session_state.setup_complete = True
            st.session_state.current_drug = selected_drug
            st.session_state.current_oil = selected_oil
            st.success("Step 1 Complete! Steps 2 and 3 are now active in the sidebar.")

    with c2:
        img, smi, mw, lp = get_structure(selected_drug)
        if img:
            st.image(img, caption=f"Chemical Structure of {selected_drug}")
            st.write(f"**MW:** {mw} | **LogP:** {lp}")

# --- CONDITIONAL PAGES ---
elif nav_choice == "Step 2: Expert Rationale":
    if not st.session_state.setup_complete:
        st.markdown("<div class='locked-msg'>🔒 Step 2 is locked. Please complete Step 1 and click 'Confirm' to unlock.</div>", unsafe_allow_html=True)
    else:
        st.header("Step 2: Evidence-Based Rationale & Outlier Detection")
        oil = st.session_state.current_oil
        drug = st.session_state.current_drug
        
        # Best Performer
        best = df[df['Oil_phase'] == oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False).iloc[0]
        
        # Outlier Detection for this specific drug
        drug_outliers = df[(df['Drug_Name'] == drug) & (df['is_outlier'] == True)]
        
        st.markdown(f"### Specific Recommendation for **{oil}**")
        st.info(f"The AI recommends **{best['Surfactant']} + {best['Co-surfactant']}** based on historical EE% of {best['Encapsulation_Efficiency_clean']:.1f}%.")
        
        if not drug_outliers.empty:
            st.markdown("#### ⚠️ Anomalous Results Detected")
            st.write(f"The following trials for {drug} showed performance significantly outside the statistical norm:")
            for i, row in drug_outliers.iterrows():
                st.markdown(f"""
                <div class="outlier-box">
                <b>System:</b> {row['Surfactant']} | <b>Observation:</b> EE% was {row['Encapsulation_Efficiency_clean']}%. 
                This is a statistical outlier compared to the rest of the {drug} formulations.
                </div>
                """, unsafe_allow_html=True)

elif nav_choice == "Step 3: Outcome Prediction":
    if not st.session_state.setup_complete:
        st.markdown("<div class='locked-msg'>🔒 Step 3 is locked. Please complete Step 1 and click 'Confirm' to unlock.</div>", unsafe_allow_html=True)
    else:
        st.header("Step 3: Optimized Formulation Prediction")
        c1, c2 = st.columns([1, 1.5])
        with c1:
            s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
            
            if st.button("🚀 Execute Prediction"):
                inputs = [[le_dict['Drug_Name'].transform([st.session_state.current_drug])[0],
                           le_dict['Oil_phase'].transform([st.session_state.current_oil])[0],
                           le_dict['Surfactant'].transform([s])[0],
                           le_dict['Co-surfactant'].transform([cs])[0]]]
                
                res = [models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
                conf = stab_model.predict_proba(inputs)[0][1] * 100
                
                metrics = [("Droplet Size", f"{res[0]:.2f} nm"), ("PDI", f"{res[1]:.3f}"), ("Zeta", f"{res[2]:.1f} mV"),
                           ("Loading", f"{res[3]:.2f} mg/mL"), ("EE %", f"{res[4]:.1f} %"), ("Stability", f"{conf:.1f} %")]
                
                for l, v in metrics:
                    st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        with c2:
            o_v, s_v = np.meshgrid(np.linspace(5, 40, 15), np.linspace(15, 65, 15))
            w_v = 100 - o_v - s_v
            mask = w_v > 0
            fig = go.Figure(data=[go.Scatter3d(x=o_v[mask], y=s_v[mask], z=w_v[mask], mode='markers',
                                               marker=dict(size=4, color=s_v[mask], colorscale='Viridis'))])
            fig.update_layout(scene=dict(xaxis_title='Oil (X)', yaxis_title='S-mix (Y)', zaxis_title='Water (Z)'))
            st.plotly_chart(fig, use_container_width=True)
