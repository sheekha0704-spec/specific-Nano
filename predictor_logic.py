import joblib
import pandas as pd

# Load the saved brains
models = joblib.load('nano_regressors.pkl')
stab_model = joblib.load('stability_model.pkl')
le_dict = joblib.load('label_encoders.pkl')

def get_encoded_val(col, val):
    try:
        return le_dict[col].transform([val])[0]
    except:
        return 0

def run_prediction(drug, oil, surf, cosurf, hlb):
    # Prepare input
    input_df = pd.DataFrame([{
        'Drug_Name_enc': get_encoded_val('Drug_Name', drug),
        'Oil_phase_enc': get_encoded_val('Oil_phase', oil),
        'Surfactant_enc': get_encoded_val('Surfactant', surf),
        'Co-surfactant_enc': get_encoded_val('Co-surfactant', cosurf),
        'HLB': hlb
    }])
    
    # Predict stability
    stability = "STABLE" if stab_model.predict(input_df)[0] == 1 else "UNSTABLE"
    
    # Predict metrics
    results = {target: models[target].predict(input_df)[0] for target in models}
    results['Stability'] = stability
    
    return results
