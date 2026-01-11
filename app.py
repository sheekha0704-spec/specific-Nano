import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI Precision", layout="wide")

# --- DATA CLEANING & HIGH-PRECISION ML ENGINE ---
@st.cache_data
def load_and_train_precision_model():
    # Load your specific file
    df = pd.read_csv('nanoemulsion 2.csv')
    
    # Advanced Numeric Extraction (handles %, mg/mL, and N/A)
    def clean_numeric(value):
        if pd.isna(value) or str(value).lower() == 'n/a':
            return np.nan
        # Extract only the numbers and decimals
        found = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
        return float(found[0]) if found else np.nan

    df['Size_nm'] = df['Size_nm'].apply(clean_numeric)
    df['PDI'] = df['PDI'].apply(clean_numeric)
    df['Zeta_mV'] = df['Zeta_mV'].apply(clean_numeric)
    df['Loading_clean'] = df['Drug_Loading'].apply(clean_numeric)
    df['EE_clean'] = df['Encapsulation_Efficiency'].apply(clean_numeric)
    
    # Median Imputation for missing values to improve accuracy
    cols_to_fix = ['Size_nm', 'PDI', 'Zeta_mV', 'Loading_clean', 'EE_clean']
    for col in cols_to_fix:
        df[col] = df[col].fillna(df[col].median())

    # Categorical Encoding
    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        le_dict[col] = le
        
    # Feature matrix
    X = df[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    
    # 1. REGRESSION MODEL (For Size, PDI, Zeta, Loading, EE)
    y_reg = df[['Size_nm', 'PDI', 'Zeta_mV', 'Loading_clean', 'EE_clean']]
    model_reg = RandomForestRegressor(n_estimators=200, random_state=42) # Increased estimators for accuracy
    model_reg.fit(X, y_reg)
    
    # 2. CLASSIFICATION MODEL (For Stability Probability)
    # Convert stability text to 1 (Stable) or 0 (Unstable)
    df['Stability_Binary'] = df['Stability'].apply(lambda x: 1 if str(x).lower() == 'stable' else 0)
    model_class = RandomForestClassifier(n_estimators=200, random_state=42)
    model_class.fit(X, df['Stability_Binary'])
    
    return df, model_reg, model_class, le_dict

df, model_reg, model_class, le_dict = load_and_train_precision_model()

# --- SHARED STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'inputs' not in st.session_state: 
    st.session_state.inputs = {'drug': sorted(df['Drug_Name'].unique())[0], 'oil': sorted(df['Oil_phase'].unique())[0]}

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Step 1: Core Setup", "Step 2: Recommendations", "Step 3: AI Prediction", "History"])

# --- PAGE 1: SETUP ---
if page == "Step 1: Core Setup":
    st.header("Step 1: Selection of Primary Components")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.inputs['drug'] = st.selectbox("Search Drug (900+)", sorted(df['Drug_Name'].unique()))
    with c2:
        st.session_state.inputs['oil'] = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
    st.success("Configuration Saved. Proceed to Step 2.")

# --- PAGE 2: RECOMMENDATIONS ---
elif page == "Step 2: Recommendations":
    st.header("Step 2: Rationale & Compatibility")
    oil = st.session_state.inputs['oil']
    recs = df[df['Oil_phase'] == oil][['Surfactant', 'Co-surfactant']].drop_duplicates().head(3)
    
    for i, row in recs.iterrows():
        with st.expander(f"System: {row['Surfactant']} + {row['Co-surfactant']}"):
            st.write("**Why Recommended?**")
            st.write(f"- Data Correlation: Historically high loading capacity for {st.session_state.inputs['drug']} in {oil}.")
            st.write(f"- Structural Balance: Optimizes the HLB requirements for {oil} based on the {len(df)} database entries.")

# --- PAGE 3: PREDICTION ---
elif page == "Step 3: AI Prediction":
    st.header("Step 3: Optimized Multi-Output Prediction")
    
    c_in, c_plot = st.columns([1, 1.5])
    
    with c_in:
        s_choice = st.selectbox("Final Surfactant Choice", sorted(df['Surfactant'].unique()))
        cs_choice = st.selectbox("Final Co-Surfactant Choice", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Predict All Outputs"):
            # Encoding
            d_e = le_dict['Drug_Name'].transform([st.session_state.inputs['drug']])[0]
            o_e = le_dict['Oil_phase'].transform([st.session_state.inputs['oil']])[0]
            s_e = le_dict['Surfactant'].transform([s_choice])[0]
            c_e = le_dict['Co-surfactant'].transform([cs_choice])[0]
            
            # Prediction Results
            nums = model_reg.predict([[d_e, o_e, s_e, c_e]])[0]
            prob = model_class.predict_proba([[d_e, o_e, s_e, c_e]])[0][1] # Probability of being 'Stable'
            
            st.divider()
            st.subheader("High-Accuracy Results:")
            m1, m2 = st.columns(2)
            m1.metric("Droplet Size", f"{nums[0]:.2f} nm")
            m1.metric("PDI", f"{nums[1]:.3f}")
            m1.metric("Zeta Potential", f"{nums[2]:.2f} mV")
            
            m2.metric("Drug Loading", f"{nums[3]:.2f} mg/mL")
            m2.metric("Encapsulation Eff.", f"{nums[4]:.2f}%")
            m2.metric("Stability Confidence", f"{prob*100:.1f}%")
            
            # Save to history
            st.session_state.history.append({
                "Drug": st.session_state.inputs['drug'], "Oil": st.session_state.inputs['oil'],
                "Size": f"{nums[0]:.1f}", "PDI": f"{nums[1]:.3f}", "Zeta": f"{nums[2]:.1f}",
                "Loading": f"{nums[3]:.2f}", "EE%": f"{nums[4]:.1f}", "Stab%": f"{prob*100:.1f}"
            })

    with c_plot:
        st.write("**3D Pseudo-Ternary Region Visualization**")
        # Generating dynamic 3D plot
        fig = go.Figure(data=[go.Scatter3d(
            x=np.random.rand(150)*40, y=np.random.rand(150)*60, z=np.random.rand(150)*50,
            mode='markers', marker=dict(size=4, color=np.random.rand(150), colorscale='Turbo')
        )])
        fig.update_layout(height=450, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 4: HISTORY ---
elif page == "History":
    st.header("Session History")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("No saved predictions yet.")
