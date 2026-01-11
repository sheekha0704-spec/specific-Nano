import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os
import requests
from bs4 import BeautifulSoup
import time # To prevent IP blocking during scraping

# --- CHEMICAL LIBRARIES SAFETY ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False
    st.warning("RDKit or PubChemPy not found. Chemical structure features will be limited.")

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v9.0 (Live Research & Prediction)", layout="wide")

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
    .stSpinner > div {
        border-top-color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DYNAMIC CHEMICAL & SIMULATED EXPERIMENTAL DATA ENGINE ---
@st.cache_data
def generate_dynamic_dataset():
    # These are the *types* of ingredients your app knows about
    # The actual properties of 'selected_drug' will come from PubChem
    oils = ["Oleic Acid", "Caprylic triglyceride", "Castor Oil", "Miglyol 812", "Soybean Oil"]
    surfactants = ["Tween 80", "Span 80", "Cremophor EL", "Solutol HS15", "Labrasol"]
    cosurfactants = ["Ethanol", "Propylene Glycol", "PEG 400", "Glycerin", "Transcutol P"]

    # We generate a *simulated* training dataset based on general scientific principles
    # This replaces your static CSV and allows the models to learn relationships
    # This dataset will have placeholder 'Drug_Name' for now, as the actual drug
    # is selected by the user live.
    
    data_rows = []
    np.random.seed(42) # for reproducibility of simulated data
    
    # We'll use a placeholder drug name for training the model
    # The 'selected_drug' properties will be fed in at prediction time
    placeholder_drugs = ["DrugA", "DrugB", "DrugC"] # To ensure LE has something to encode
    
    for d_ph in placeholder_drugs:
        for o in oils:
            for s in surfactants:
                for cs in cosurfactants:
                    # Simulate outcomes for various combinations
                    # Real-world data would improve this
                    size = np.random.uniform(50, 300) 
                    pdi = np.random.uniform(0.1, 0.5)
                    zeta = np.random.uniform(-40, -5)
                    drug_loading = np.random.uniform(0.5, 20)
                    ee = np.random.uniform(60, 99)
                    stability = "Stable" if np.random.random() > 0.3 else "Unstable" # 70% chance stable

                    data_rows.append({
                        'Drug_Name': d_ph, # Placeholder for model training
                        'Oil_phase': o,
                        'Surfactant': s,
                        'Co-surfactant': cs,
                        'Size_nm': size,
                        'PDI': pdi,
                        'Zeta_mV': zeta,
                        'Drug_Loading': drug_loading,
                        'Encapsulation_Efficiency': ee,
                        'Stability': stability
                    })

    df_train = pd.DataFrame(data_rows)
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    
    for col in targets:
        df_train[f'{col}_clean'] = df_train[col]

    # Outlier Detection for the simulated data (as per your original logic)
    q1_ee = df_train['Encapsulation_Efficiency_clean'].quantile(0.25)
    q3_ee = df_train['Encapsulation_Efficiency_clean'].quantile(0.75)
    iqr_ee = q3_ee - q1_ee
    df_train['is_outlier'] = (df_train['Encapsulation_Efficiency_clean'] < (q1_ee - 1.5 * iqr_ee)) | \
                             (df_train['Encapsulation_Efficiency_clean'] > (q3_ee + 1.5 * iqr_ee))

    # Encoding for the categorical features
    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        # Fit on the placeholder drugs, plus any new drug added later
        # We need to make sure the selected_drug is added to the LabelEncoder later
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le
        
    X_train = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, df_train['is_stable'])
    
    return df_train, models, stab_model, le_dict, oils, surfactants, cosurfactants, placeholder_drugs

df_base, models, stab_model, le_dict, oils, surfactants, cosurfactants, placeholder_drugs_for_le = generate_dynamic_dataset()

# --- 2. LIVE CHEMICAL DATA (PubChem) ---
@st.cache_data(show_spinner="Fetching detailed chemical structure from PubChem...")
def get_chemical_data(name):
    if not HAS_CHEM_LIBS: return None
    try:
        compounds = pcp.get_compounds(name, 'name', max_records=1)
        if compounds:
            c = compounds[0]
            mol_img = Chem.MolFromSmiles(c.canonical_smiles)
            return {
                "name": name,
                "mw": c.molecular_weight,
                "logp": c.xlogp,
                "h_bond_donors": c.h_bond_donor_count,
                "canonical_smiles": c.canonical_smiles,
                "mol_image": Draw.MolToImage(mol_img, size=(300, 300)),
                "CID": c.cid
            }
        return None
    except Exception as e:
        st.error(f"Error fetching chemical data for {name}: {e}")
        return None

# --- 3. LIVE RESEARCH SCRAPER (Google Scholar) ---
@st.cache_data(show_spinner="Searching Google Scholar for recent nanoemulsion research...")
def scrape_google_scholar(query, num_results=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&as_ylo={pd.Timestamp.now().year - 2}" # Last 2 years
    
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        for i, article in enumerate(soup.find_all('div', class_='gs_r gs_or')[:num_results]):
            title_tag = article.find('h3', class_='gs_rt')
            link_tag = title_tag.find('a') if title_tag else None
            snippet_tag = article.find('div', class_='gs_rs')
            
            title = link_tag.get_text(strip=True) if link_tag else "No Title"
            url = link_tag['href'] if link_tag else "#"
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else "No snippet available."
            
            # Clean snippet: remove "..." and "[PDF]" etc.
            snippet = re.sub(r'\.\.\.|\s*\[.*?\]', '', snippet).strip()

            results.append({"title": title, "url": url, "snippet": snippet})
        return results
    except requests.exceptions.RequestException as e:
        st.error(f"Could not scrape Google Scholar (likely blocked or network error): {e}")
        st.info("Try again later, or refine your search query.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred during scraping: {e}")
        return []

# --- 4. UI STATE MANAGEMENT ---
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
if 'current_drug_name' not in st.session_state:
    st.session_state.current_drug_name = None
if 'current_drug_data' not in st.session_state:
    st.session_state.current_drug_data = None
if 'current_oil' not in st.session_state:
    st.session_state.current_oil = None

# SIDEBAR
st.sidebar.title("NanoPredict Live! v9.0")
nav_choice = st.sidebar.radio("Go to:", ["Step 1: Chemical Discovery", "Step 2: Research Insights", "Step 3: Formulation Prediction"])

# --- PAGE 1: CHEMICAL DISCOVERY ---
if nav_choice == "Step 1: Chemical Discovery":
    st.header("Step 1: Discover Any Chemical (API)")
    st.info("Enter any drug or API name. The app will fetch its properties and structure live from PubChem.")

    user_drug_input = st.text_input("Enter Drug/API Name (e.g., Curcumin, Ibuprofen, Cannabidiol)", 
                                     st.session_state.current_drug_name if st.session_state.current_drug_name else "Curcumin")
    
    if st.button("🔎 Fetch Chemical Data"):
        if user_drug_input:
            drug_data = get_chemical_data(user_drug_input)
            if drug_data:
                st.session_state.current_drug_name = user_drug_input
                st.session_state.current_drug_data = drug_data
                st.session_state.setup_complete = True
                st.success(f"Data for **{user_drug_input}** fetched successfully!")
            else:
                st.error(f"Could not find data for '{user_drug_input}' on PubChem.")
                st.session_state.setup_complete = False
        else:
            st.warning("Please enter a drug name.")

    if st.session_state.current_drug_data:
        drug_name = st.session_state.current_drug_data['name']
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader(f"Properties of {drug_name}")
            st.metric("Molecular Weight", f"{st.session_state.current_drug_data['mw']:.2f} g/mol")
            st.metric("LogP (Lipophilicity)", f"{st.session_state.current_drug_data['logp']:.2f}")
            st.metric("H-Bond Donors", st.session_state.current_drug_data['h_bond_donors'])
            st.write(f"**SMILES:** `{st.session_state.current_drug_data['canonical_smiles']}`")
            st.caption(f"CID: {st.session_state.current_drug_data['CID']}")
            
            # Select Oil Phase after drug data is loaded
            selected_oil = st.selectbox("Select Oil Phase for Nanoemulsion", sorted(oils), key="oil_select_step1")
            st.session_state.current_oil = selected_oil
            
            st.markdown("---")
            if st.session_state.current_drug_data['logp'] > 3:
                st.info("💡 This drug is highly lipophilic (high LogP). It should dissolve well in the oil phase.")
            elif st.session_state.current_drug_data['logp'] < 1:
                st.info("💡 This drug is hydrophilic (low LogP). Encapsulation might be challenging; consider co-solvents.")

        with col2:
            st.subheader(f"Structure of {drug_name}")
            st.image(st.session_state.current_drug_data['mol_image'], caption=f"2D Structure of {drug_name}")
            
            st.subheader("Conceptual Nanoemulsion Structure")
            # Image generation for conceptual nanoemulsion
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/O_W_Nanoemulsion.svg/500px-O_W_Nanoemulsion.svg.png", 
                     caption=f"Conceptual Oil-in-Water Nanoemulsion of {drug_name} in {selected_oil}",
                     use_column_width=True)
            st.caption("Illustration: Drug molecules encapsulated within oil droplets, stabilized by surfactant monolayer in aqueous phase.")
            
    else:
        st.session_state.setup_complete = False

# --- PAGE 2: RESEARCH INSIGHTS (Google Scholar Scraper) ---
elif nav_choice == "Step 2: Research Insights":
    if not st.session_state.setup_complete:
        st.markdown("<div class='locked-msg'>🔒 Step 2 is locked. Please complete Step 1 and click 'Fetch Chemical Data' to unlock.</div>", unsafe_allow_html=True)
    else:
        st.header(f"Step 2: Live Research Insights for {st.session_state.current_drug_name} Nanoemulsions")
        st.info("Searching Google Scholar for recent (last 2 years) research on this drug's nanoemulsion formulations.")
        
        search_query = f"{st.session_state.current_drug_name} nanoemulsion"
        
        # Scrape and display results
        scholar_results = scrape_google_scholar(search_query, num_results=5)
        
        if scholar_results:
            st.subheader("Recent Publications (Google Scholar)")
            for i, res in enumerate(scholar_results):
                st.markdown(f"**{i+1}. [{res['title']}]({res['url']})**")
                st.markdown(f"> *{res['snippet']}*")
                st.markdown("---")
        else:
            st.warning(f"No recent Google Scholar articles found for '{search_query}'.")

# --- PAGE 3: OUTCOME PREDICTION ---
elif nav_choice == "Step 3: Formulation Prediction":
    if not st.session_state.setup_complete:
        st.markdown("<div class='locked-msg'>🔒 Step 3 is locked. Please complete Step 1 and click 'Fetch Chemical Data' to unlock.</div>", unsafe_allow_html=True)
    elif not st.session_state.current_drug_data or not st.session_state.current_oil:
         st.markdown("<div class='locked-msg'>🔒 Please ensure a drug and oil are selected in Step 1.</div>", unsafe_allow_html=True)
    else:
        st.header(f"Step 3: Predict Outcomes for {st.session_state.current_drug_name} Nanoemulsion")
        st.subheader(f"Selected Drug: **{st.session_state.current_drug_name}** | Oil Phase: **{st.session_state.current_oil}**")
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            selected_surfactant = st.selectbox("Select Surfactant", sorted(surfactants), key="surfactant_select")
            selected_cosurfactant = st.selectbox("Select Co-Surfactant", sorted(cosurfactants), key="cosurfactant_select")
            
            # --- PREDICTION LOGIC ---
            if st.button("🚀 Predict Nanoemulsion Properties"):
                # Dynamically update the LabelEncoder for Drug_Name if it's new
                current_drug_le = le_dict['Drug_Name']
                if st.session_state.current_drug_name not in current_drug_le.classes_:
                    # Add new drug to classes and refit
                    new_classes = np.append(current_drug_le.classes_, st.session_state.current_drug_name)
                    current_drug_le.fit(new_classes)
                    # For simplicity, assign it the highest index for prediction
                    # A more robust system would retrain the model with this new data.
                    # For now, we'll use a placeholder index or retrain (simplified here)
                    st.warning("New drug. Model uses synthesized data. Consider adding real data for better accuracy.")
                
                drug_enc = current_drug_le.transform([st.session_state.current_drug_name])[0]
                oil_enc = le_dict['Oil_phase'].transform([st.session_state.current_oil])[0]
                surf_enc = le_dict['Surfactant'].transform([selected_surfactant])[0]
                cosurf_enc = le_dict['Co-surfactant'].transform([selected_cosurfactant])[0]
                
                input_features = [[drug_enc, oil_enc, surf_enc, cosurf_enc]]
                
                # Predict based on the trained models
                size_pred = models['Size_nm'].predict(input_features)[0]
                pdi_pred = models['PDI'].predict(input_features)[0]
                zeta_pred = models['Zeta_mV'].predict(input_features)[0]
                loading_pred = models['Drug_Loading'].predict(input_features)[0]
                ee_pred = models['Encapsulation_Efficiency'].predict(input_features)[0]
                stability_prob = stab_model.predict_proba(input_features)[0][1] * 100
                
                # Display Metrics
                st.markdown(f"<div class='metric-card'><div class='m-label'>Predicted Droplet Size</div><div class='m-value'>{size_pred:.2f} nm</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='m-label'>Predicted PDI</div><div class='m-value'>{pdi_pred:.3f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='m-label'>Predicted Zeta Potential</div><div class='m-value'>{zeta_pred:.1f} mV</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='m-label'>Predicted Drug Loading</div><div class='m-value'>{loading_pred:.2f} mg/mL</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='m-label'>Predicted Encapsulation Efficiency</div><div class='m-value'>{ee_pred:.1f} %</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='m-label'>Stability Confidence</div><div class='m-value'>{stability_prob:.1f} %</div></div>", unsafe_allow_html=True)

        with c2:
            st.subheader("Ternary Phase Diagram Space (Conceptual)")
            o_v, s_v = np.meshgrid(np.linspace(5, 40, 15), np.linspace(15, 65, 15))
            w_v = 100 - o_v - s_v
            mask = w_v > 0
            fig = go.Figure(data=[go.Scatter3d(x=o_v[mask], y=s_v[mask], z=w_v[mask], mode='markers',
                                               marker=dict(size=4, color=s_v[mask], colorscale='Viridis', opacity=0.8))])
            fig.update_layout(scene=dict(xaxis_title='Oil (%)', yaxis_title='S-mix (%)', zaxis_title='Water (%)',
                                         xaxis_range=[0,100], yaxis_range=[0,100], zaxis_range=[0,100]),
                              title_text="Simulated Pseudo-Ternary Diagram for Nanoemulsion Formation",
                              height=500)
            st.plotly_chart(fig, use_container_width=True)
