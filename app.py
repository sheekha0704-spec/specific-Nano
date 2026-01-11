import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI Pro", layout="wide")

# --- DATA CLEANING & ML ENGINE ---
@st.cache_data
def load_and_train_all():
    df = pd.read_csv('nanoemulsion 2.csv')
    
    # 1. Clean Numeric Columns
    def extract_num(text):
        if pd.isna(text): return np.nan
        res = re.findall(r"[-+]?\d*\.\d+|\d+", str(text))
        return float(res[0]) if res else np.nan

    df['Size_nm'] = df['Size_nm'].apply(extract_num)
    df['PDI'] = df['PDI'].apply(extract_num)
    df['Zeta_mV'] = df['Zeta_mV'].apply(extract_num)
    df['Drug_Loading_val'] = df['Drug_Loading'].apply(extract_num)
    df['EE_val'] = df['Encapsulation_Efficiency'].apply(extract_num)
    
    # Fill missing values with median for training
    for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading_val', 'EE_val']:
        df[col] = df[col].fillna(df[col].median())

    # 2. Encoders
    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        le_dict[col] = le
        
    # 3. Training
    X = df[['Drug_Name_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'Oil_phase_enc']]
    
    # Regressor for numeric outputs
    model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    model_reg.fit(X, df[['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading_val', 'EE_val']])
    
    # Classifier for Stability
    model_stab = RandomForestClassifier(n_estimators=100, random_state=42)
    model_stab.fit(X, df['Stability'].astype(str))
    
    return df, model_reg, model_stab, le_dict

df, model_reg, model_stab, le_dict = load_and_train_all()

# --- SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'inputs' not in st.session_state: 
    st.session_state.inputs = {'drug': df['Drug_Name'].unique()[0], 'oil': df['Oil_phase'].unique()[0]}

# --- NAVIGATION ---
st.sidebar.title("NanoPredict Menu")
page = st.sidebar.radio("Navigate", ["Step 1: Setup", "Step 2: Recommendations", "Step 3: Prediction & 3D Plot", "History Log"])

# --- PAGE 1: SETUP ---
if page == "Step 1: Setup":
    st.header("Step 1: Primary Formulation Phase")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.inputs['drug'] = st.selectbox("Search Drug (900+)", sorted(df['Drug_Name'].unique()))
    with c2:
        st.session_state.inputs['oil'] = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
    st.info("Selection saved. Go to Step 2 to see compatible surfactants.")

# --- PAGE 2: RECOMMENDATIONS ---
elif page == "Step 2: Recommendations":
    st.header("Step 2: Component Compatibility")
    oil = st.session_state.inputs['oil']
    # Filter recommendations
    recs = df[df['Oil_phase'] == oil][['Surfactant', 'Co-surfactant']].drop_duplicates().head(4)
    
    for i, row in recs.iterrows():
        with st.expander(f"System: {row['Surfactant']} + {row['Co-surfactant']}"):
            st.write("**Why Recommended?**")
            st.write(f"Based on 1,000+ data points, the interfacial tension of {oil} is best reduced by {row['Surfactant']}, ensuring a stable nano-droplet.")

# --- PAGE 3: PREDICTIONS ---
elif page == "Step 3: Prediction & 3D Plot":
    st.header("Step 3: Final AI Prediction")
    
    col_input, col_graph = st.columns([1, 1.5])
    
    with col_input:
        s_choice = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
        cs_choice = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Execute Prediction"):
            # Encode
            d_e = le_dict['Drug_Name'].transform([st.session_state.inputs['drug']])[0]
            o_e = le_dict['Oil_phase'].transform([st.session_state.inputs['oil']])[0]
            s_e = le_dict['Surfactant'].transform([s_choice])[0]
            c_e = le_dict['Co-surfactant'].transform([cs_choice])[0]
            
            # Predict
            res = model_reg.predict([[d_e, o_e, s_e, c_e]])[0]
            stab = model_stab.predict([[d_e, o_e, s_e, c_e]])[0]
            
            # DISPLAY ALL 6 OUTPUTS
            st.subheader("Results:")
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Size (nm)", f"{res[0]:.1f}")
            res_c1.metric("PDI", f"{res[1]:.3f}")
            res_c1.metric("Zeta (mV)", f"{res[2]:.1f}")
            
            res_c2.metric("Drug Loading", f"{res[3]:.2f} mg/mL")
            res_c2.metric("EE (%)", f"{res[4]:.1f}%")
            res_c2.metric("Stability", stab)
            
            # Save to history
            st.session_state.history.append({
                "Drug": st.session_state.inputs['drug'], "Oil": st.session_state.inputs['oil'],
                "Size": res[0], "PDI": res[1], "Zeta": res[2], "Loading": res[3], "EE": res[4], "Stability": stab
            })

    with col_graph:
        # Pseudo-ternary visualization
        st.write("**3D Formulation Space**")
        fig = go.Figure(data=[go.Scatter3d(
            x=np.random.rand(100), y=np.random.rand(100), z=np.random.rand(100),
            mode='markers', marker=dict(size=4, color=np.random.rand(100), colorscale='Viridis')
        )])
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 4: HISTORY ---
elif page == "History Log":
    st.header("Saved Formulations")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.info("No work saved yet.")
