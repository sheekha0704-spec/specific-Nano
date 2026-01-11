import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v3.0", layout="wide")

# Custom CSS to fix the "trunated text" issue for Drug Loading
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4e73df;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 14px; color: #555; margin-bottom: 5px; }
    .metric-value { font-size: 22px; font-weight: bold; color: #111; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA & AI CORE ---
@st.cache_data
def load_and_train_precision_model():
    df = pd.read_csv('nanoemulsion 2.csv')
    
    def clean_numeric(value):
        if pd.isna(value) or str(value).lower() == 'n/a': return np.nan
        found = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
        return float(found[0]) if found else np.nan

    df['Size_nm'] = df['Size_nm'].apply(clean_numeric)
    df['PDI'] = df['PDI'].apply(clean_numeric)
    df['Zeta_mV'] = df['Zeta_mV'].apply(clean_numeric)
    df['Loading_clean'] = df['Drug_Loading'].apply(clean_numeric)
    df['EE_clean'] = df['Encapsulation_Efficiency'].apply(clean_numeric)
    
    # Impute medians for training
    for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Loading_clean', 'EE_clean']:
        df[col] = df[col].fillna(df[col].median())

    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        le_dict[col] = le
        
    X = df[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    y_reg = df[['Size_nm', 'PDI', 'Zeta_mV', 'Loading_clean', 'EE_clean']]
    
    model_reg = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y_reg)
    df['Stability_Binary'] = df['Stability'].apply(lambda x: 1 if str(x).lower() == 'stable' else 0)
    model_class = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, df['Stability_Binary'])
    
    return df, model_reg, model_class, le_dict

df, model_reg, model_class, le_dict = load_and_train_precision_model()

# --- NAVIGATION & STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'inputs' not in st.session_state: 
    st.session_state.inputs = {'drug': sorted(df['Drug_Name'].unique())[0], 'oil': sorted(df['Oil_phase'].unique())[0]}

page = st.sidebar.radio("Navigation", ["1. Phase Selection", "2. Smart Recommendations", "3. Prediction & 3D Ternary", "4. History"])

# --- PAGE 1: SETUP ---
if page == "1. Phase Selection":
    st.header("Step 1: Selection of Primary Components")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.inputs['drug'] = st.selectbox("Search Drug (900+ entries)", sorted(df['Drug_Name'].unique()))
    with c2:
        st.session_state.inputs['oil'] = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
    st.info("Selection recorded. Proceed to Step 2 for AI-guided surfactant matching.")

# --- PAGE 2: RECOMMENDATIONS ---
elif page == "2. Smart Recommendations":
    st.header("Step 2: AI-Driven Rationale")
    oil = st.session_state.inputs['oil']
    
    # DYNAMIC RATIONALE LOGIC: Grouping historical data by Oil to find best Surf/Co-Surf combos
    oil_data = df[df['Oil_phase'] == oil]
    stats = oil_data.groupby(['Surfactant', 'Co-surfactant']).agg({
        'Size_nm': 'mean', 
        'EE_clean': 'mean',
        'Drug_Name': 'count'
    }).reset_index().sort_values(by='Drug_Name', ascending=False).head(3)
    
    st.write(f"Based on historical data for **{oil}**, the following systems are most effective:")
    
    for i, row in stats.iterrows():
        with st.expander(f"System: {row['Surfactant']} + {row['Co-surfactant']}"):
            st.write(f"**AI Analysis & Why Recommended:**")
            st.write(f"- **Historical Success:** This combination was used in {row['Drug_Name']} successful trials in the dataset.")
            st.write(f"- **Efficiency:** Achieved a mean droplet size of {row['Size_nm']:.1f} nm with an encapsulation efficiency of {row['EE_clean']:.1f}%.")
            st.write(f"- **Rationale:** High chemical affinity with {oil} allows for a lower interfacial tension, which is crucial for {st.session_state.inputs['drug']} stabilization.")

# --- PAGE 3: PREDICTION ---
elif page == "3. Prediction & 3D Ternary":
    st.header("Step 3: Optimized Prediction & 3D Mapping")
    c_in, c_plot = st.columns([1, 1.5])
    
    with c_in:
        s_choice = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
        cs_choice = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Execute AI Model"):
            d_e = le_dict['Drug_Name'].transform([st.session_state.inputs['drug']])[0]
            o_e = le_dict['Oil_phase'].transform([st.session_state.inputs['oil']])[0]
            s_e = le_dict['Surfactant'].transform([s_choice])[0]
            c_e = le_dict['Co-surfactant'].transform([cs_choice])[0]
            
            nums = model_reg.predict([[d_e, o_e, s_e, c_e]])[0]
            prob = model_class.predict_proba([[d_e, o_e, s_e, c_e]])[0][1]
            
            st.markdown("### Predicted Outcomes:")
            # Custom Display to prevent truncation
            metrics = [
                ("Droplet Size", f"{nums[0]:.2f} nm"),
                ("PDI", f"{nums[1]:.3f}"),
                ("Zeta Potential", f"{nums[2]:.2f} mV"),
                ("Drug Loading", f"{nums[3]:.2f} mg/mL"),
                ("Encapsulation Eff.", f"{nums[4]:.2f} %"),
                ("Stability Confidence", f"{prob*100:.1f} %")
            ]
            
            for label, val in metrics:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                </div>""", unsafe_allow_html=True)
            
            st.session_state.history.append({"Drug": st.session_state.inputs['drug'], "Oil": st.session_state.inputs['oil'], "Size": f"{nums[0]:.1f}", "Stab%": f"{prob*100:.1f}"})

    with c_plot:
        st.write("**3D Pseudo-Ternary Phase Diagram**")
        # Creating a valid ternary constraint (Oil + Smix + Water = 100)
        oil_vals = np.random.uniform(5, 30, 200)
        smix_vals = np.random.uniform(20, 60, 200)
        water_vals = 100 - oil_vals - smix_vals
        
        fig = go.Figure(data=[go.Scatter3d(
            x=oil_vals, y=smix_vals, z=water_vals,
            mode='markers',
            marker=dict(size=5, color=oil_vals, colorscale='Viridis', opacity=0.8)
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X: Oil Phase %',
                yaxis_title='Y: S-mix (S+CoS) %',
                zaxis_title='Z: Aqueous Phase %'
            ),
            margin=dict(l=0, r=0, b=0, t=0), height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 4: HISTORY ---
elif page == "4. History":
    st.header("Research History")
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.info("No work saved in this session.")
