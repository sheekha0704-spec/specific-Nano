import streamlit as st
from predictor_logic import run_prediction, le_dict # Import our engine

st.set_page_config(page_title="NanoPredict AI v2.0", layout="wide")

# ... (Keep your custom CSS and Sidebar logic here) ...

# Inside Step 6: Analysis
if st.button("Generate AI Report"):
    results = run_prediction(
        st.session_state.get('drug_name', 'None'),
        st.session_state.get('oil_choice', 'None'),
        st.session_state.get('s_final', 'None'),
        st.session_state.get('cs_final', 'None'),
        12.0 # HLB
    )
    
    # Display stability
    bg = "#d4edda" if results['Stability'] == "STABLE" else "#f8d7da"
    st.markdown(f'<div class="status-box" style="background-color: {bg};">{results["Stability"]}</div>', unsafe_allow_html=True)

    # Display Metrics using your card style
    p_cols = st.columns(4)
    for i, target in enumerate(['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']):
        with p_cols[i]:
            st.markdown(f"<div class='metric-card'><div class='m-label'>{target}</div><div class='m-value'>{results[target]:.2f}</div></div>", unsafe_allow_html=True)
