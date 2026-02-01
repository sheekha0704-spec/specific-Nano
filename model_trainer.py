import pandas as pd
import joblib
import re
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_and_save():
    df = pd.read_csv('nanoemulsion 2.csv')
    
    # 1. Cleaning & Feature Engineering
    hlb_map = {'Tween 80': 15.0, 'Tween 20': 16.7, 'Span 80': 4.3, 'Cremophor EL': 13.5, 'Labrasol': 12.0}
    df['HLB'] = df['Surfactant'].map(hlb_map).fillna(12.0)
    
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].fillna("Unknown").astype(str))
        le_dict[col] = le

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    features = ['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'HLB']
    X = df[[f'{c}_enc' if c in cat_cols else c for c in features]]

    # 2. Train Models
    trained_models = {}
    for col in targets:
        clean_target = df[col].apply(get_num)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, clean_target)
        trained_models[target] = model

    # Stability Model
    df['is_stable'] = df.get('Stability', pd.Series(['stable']*len(df))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(random_state=42).fit(X, df['is_stable'])

    # 3. EXPORT ASSETS
    joblib.dump(trained_models, 'nano_regressors.pkl')
    joblib.dump(stab_model, 'stability_model.pkl')
    joblib.dump(le_dict, 'label_encoders.pkl')
    print("Training Complete. 3 files generated.")

if __name__ == "__main__":
    train_and_save()
