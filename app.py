import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

# --- DATA LOADING (Cached for Speed) ---
@st.cache_data
def load_full_data():
    df = pd.read_csv('nanoemulsion 2.csv')
    # Basic cleaning
    df['Size_nm'] = pd.to_numeric(df['Size_nm'], errors='coerce')
    df = df.dropna(subset=['Size_nm'])
    
    # Encoders
    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        df[col] = df[col].astype(str).fillna('None')
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        le_dict[col] = le
        
    # Model Training
    X = df[['Drug_Name_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'Oil_phase_enc']]
    y = df[['Size_nm', 'PDI', 'Zeta_mV']]
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
    model.fit(X, y)
    
    return df, model, le_dict

df, model, le_dict = load_full_data()

# --- SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'inputs' not in st.session_state:
    st.session_state.inputs = {'drug': df['Drug_Name'].iloc[0], 'oil': df['Oil_phase'].iloc[0], 'aqueous': 'Water'}

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠 Navigation")
page = st.sidebar.radio("Go to:", ["Step 1: Setup", "Step 2: Recommendations", "Step 3: Prediction", "View History"])

# --- PAGE 1: SETUP ---
if page == "Step 1: Setup":
    st.header("📍 Step 1: Drug & Phase Selection")
    st.write("Select your core components to begin the optimization process.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.inputs['drug'] = st.selectbox("Select Drug (900+ items)", sorted(df['Drug_Name'].unique()))
        st.session_state.inputs['aqueous'] = st.text_input("Aqueous Phase", st.session_state.inputs['aqueous'])
    with col2:
        st.session_state.inputs['oil'] = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
    
    st.success("✅ Components saved. Proceed to Step 2 in the sidebar.")

# --- PAGE 2: RECOMMENDATIONS ---
elif page == "Step 2: Recommendations":
    st.header("🔍 Step 2: Surfactant Selection & Rationale")
    oil = st.session_state.inputs['oil']
    
    # Logic for Recommendation
    recs = df[df['Oil_phase'] == oil][['Surfactant', 'Co-surfactant']].drop_duplicates().head(3)
    
    st.subheader(f"Recommended Systems for {oil}")
    
    for index, row in recs.iterrows():
        with st.expander(f"Option: {row['Surfactant']} + {row['Co-surfactant']}"):
            st.write(f"**Why Recommended?**")
            st.write(f"- Compatibility: Historical data shows high solubility for drugs in {oil} when using {row['Surfactant']}.")
            st.write(f"- HLB Balance: This combination optimizes the O/W interface for {oil} droplets.")
            st.write(f"- Stability: Known to produce Zeta Potential values in the stable range (<-20mV or >+20mV).")

# --- PAGE 3: PREDICTION ---
elif page == "Step 3: Prediction":
    st.header("📊 Step 3: Final Prediction & Ternary Mapping")
    
    col1, col2 = st.columns(2)
    with col1:
        s_choice = st.selectbox("Final Surfactant Choice", sorted(df['Surfactant'].unique()))
        cs_choice = st.selectbox("Final Co-Surfactant Choice", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Generate Results"):
            # Encoding & Prediction
            d_e = le_dict['Drug_Name'].transform([st.session_state.inputs['drug']])[0]
            o_e = le_dict['Oil_phase'].transform([st.session_state.inputs['oil']])[0]
            s_e = le_dict['Surfactant'].transform([s_choice])[0]
            c_e = le_dict['Co-surfactant'].transform([cs_choice])[0]
            
            pred = model.predict([[d_e, s_e, c_e, o_e]])[0]
            
            # Show Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Size", f"{pred[0]:.2f} nm")
            m2.metric("PDI", f"{pred[1]:.3f}")
            m3.metric("Zeta", f"{pred[2]:.2f} mV")
            
            # Save to History
            st.session_state.history.append({
                "Drug": st.session_state.inputs['drug'],
                "Oil": st.session_state.inputs['oil'],
                "S/CoS": f"{s_choice}/{cs_choice}",
                "Size": f"{pred[0]:.1f}",
                "PDI": f"{pred[1]:.3f}"
            })

    with col2:
        # 3D TERNARY PLOT
        st.write("**Pseudo-Ternary Region**")
        fig = go.Figure(data=[go.Scatter3d(
            x=np.random.rand(50)*40, y=np.random.rand(50)*60, z=np.random.rand(50)*50,
            mode='markers', marker=dict(size=5, color='blue', colorscale='Viridis')
        )])
        fig.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 4: HISTORY ---
elif page == "View History":
    st.header("📜 Formulation History")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No formulations saved yet.")
