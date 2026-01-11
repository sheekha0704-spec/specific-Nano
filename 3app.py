import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re, os
from streamlit.components.v1 import html

# -------------------- OPTIONAL CHEM LIBS --------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except:
    HAS_CHEM_LIBS = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="NanoPredict AI v9.1", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.metric-card {
    background:#ffffff; padding:20px; border-radius:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    border-left:8px solid #00d8d6; margin-bottom:16px;
}
.m-label { font-size:13px; color:#555; font-weight:600; }
.m-value { font-size:26px; font-weight:800; }
.model-box {
    background:#1e272e; color:#fff; padding:22px;
    border-radius:14px; border-top:4px solid #00d8d6;
}
.locked-msg {
    text-align:center; padding:60px;
    font-style:italic; color:#a0aec0;
}
</style>
""", unsafe_allow_html=True)

# -------------------- DATA + MODEL --------------------
@st.cache_data
def load_and_prep():
    if not os.path.exists("nanoemulsion 2.csv"):
        st.error("❌ nanoemulsion 2.csv missing")
        st.stop()

    df = pd.read_csv("nanoemulsion 2.csv")

    def num(x):
        if pd.isna(x): return np.nan
        m = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(m[0]) if m else np.nan

    targets = ['Size_nm','PDI','Zeta_mV','Drug_Loading','Encapsulation_Efficiency']
    for t in targets:
        df[f"{t}_c"] = df[t].apply(num)

    df = df.dropna(subset=[f"{t}_c" for t in targets])
    for c in ['Drug_Name','Oil_phase','Surfactant','Co-surfactant']:
        df[c] = df[c].fillna("Unknown")

    enc = {}
    for c in ['Drug_Name','Oil_phase','Surfactant','Co-surfactant']:
        le = LabelEncoder()
        df[c+"_e"] = le.fit_transform(df[c])
        enc[c] = le

    X = df[['Drug_Name_e','Oil_phase_e','Surfactant_e','Co-surfactant_e']]
    models = {t: GradientBoostingRegressor(n_estimators=300, random_state=42)
              .fit(X, df[f"{t}_c"]) for t in targets}

    stab = (df['Stability'].str.lower().str.contains('stable')).astype(int)
    stab_model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, stab)

    return df, models, stab_model, enc

df, models, stab_model, enc = load_and_prep()

# -------------------- STRUCTURE --------------------
@st.cache_data
def mol_img(name):
    if not HAS_CHEM_LIBS: return None
    try:
        c = pcp.get_compounds(name,'name')[0]
        m = Chem.MolFromSmiles(c.canonical_smiles)
        return Draw.MolToImage(m, size=(280,280))
    except: return None

# -------------------- PREMIUM ANIMATION --------------------
def nano_animation(size, pdi, zeta, drug, oil):
    breakup = min(0.12 + pdi/4, 0.28)
    radius = max(5, min(size/6, 24))

    return f"""
<canvas id="nano"></canvas>
<script>
const c=document.getElementById("nano"),x=c.getContext("2d");
c.width=620;c.height=340;let d=[];
class P{{
constructor(r){{this.x=c.width/2;this.y=c.height/2;
this.r=r;this.dx=(Math.random()-.5)*1.6;this.dy=(Math.random()-.5)*1.6}}
m(){{this.x+=this.dx;this.y+=this.dy;
if(this.x<0||this.x>c.width)this.dx*=-1;
if(this.y<0||this.y>c.height)this.dy*=-1}}
b(){{if(this.r>{radius}&&Math.random()< {breakup}){{
d.push(new P(this.r*.65),new P(this.r*.65));this.r*=.55}}}
dr(){{x.beginPath();x.arc(this.x,this.y,this.r,0,7);
x.fillStyle="rgba(0,216,214,0.7)";
x.shadowBlur=10;x.shadowColor="#00d8d6";x.fill();}}
}}
function i(){{d=[];for(let i=0;i<6;i++)d.push(new P({radius*2}))}}
function a(){{x.fillStyle="rgba(15,32,39,.35)";
x.fillRect(0,0,c.width,c.height);
d.forEach(p=>{{p.m();p.b();p.dr();}});
requestAnimationFrame(a)}}
i();a();
</script>
<div style="text-align:center;color:#00d8d6;font-size:12px">
{drug} | {oil} | Size {size:.1f} nm | PDI {pdi:.2f} | Zeta {zeta:.1f} mV
</div>
"""

# -------------------- SIDEBAR --------------------
st.sidebar.title("NanoPredict Controls")
page = st.sidebar.radio("Navigate",["Step 1: Chemical Setup",
                                   "Step 2: Expert Rationale",
                                   "Step 3: Prediction & Nanomodel"])

if "ok" not in st.session_state:
    st.session_state.ok = False

# -------------------- STEP 1 --------------------
if page=="Step 1: Chemical Setup":
    st.header("🧪 Chemical & Lipid Setup")
    c1,c2 = st.columns(2)
    with c1:
        drug = st.selectbox("API", sorted(df['Drug_Name'].unique()))
        oil = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
        if st.button("Unlock System"):
            st.session_state.ok=True
            st.session_state.drug=drug
            st.session_state.oil=oil
            st.success("Unlocked")
    with c2:
        img = mol_img(drug)
        if img: st.image(img, caption=f"{drug} structure")

# -------------------- STEP 2 --------------------
elif page=="Step 2: Expert Rationale":
    if not st.session_state.ok:
        st.markdown("<div class='locked-msg'>Locked</div>",unsafe_allow_html=True)
    else:
        st.header("🔬 Scientific Rationale")
        best = df[df['Oil_phase']==st.session_state.oil]\
               .sort_values("Encapsulation_Efficiency_c",ascending=False).iloc[0]
        st.info(f"Optimal surfactant system: **{best['Surfactant']} + {best['Co-surfactant']}**")
        st.write(f"Reported EE ≈ **{best['Encapsulation_Efficiency_c']:.1f}%**")

# -------------------- STEP 3 --------------------
elif page=="Step 3: Prediction & Nanomodel":
    if not st.session_state.ok:
        st.markdown("<div class='locked-msg'>Locked</div>",unsafe_allow_html=True)
    else:
        st.header("🚀 Prediction & Nanoemulsion Formation")
        c1,c2 = st.columns([1,1.25])

        with c1:
            s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            cs = st.selectbox("Co-surfactant", sorted(df['Co-surfactant'].unique()))
            if st.button("Execute Simulation"):
                X = [[enc['Drug_Name'].transform([st.session_state.drug])[0],
                      enc['Oil_phase'].transform([st.session_state.oil])[0],
                      enc['Surfactant'].transform([s])[0],
                      enc['Co-surfactant'].transform([cs])[0]]]
                res = [models[t].predict(X)[0] for t in
                       ['Size_nm','PDI','Zeta_mV','Drug_Loading','Encapsulation_Efficiency']]

                for lab,val in zip(
                    ["Size (nm)","Loading","EE %"],
                    [f"{res[0]:.1f}",f"{res[3]:.2f}",f"{res[4]:.1f}"]):
                    st.markdown(
                        f"<div class='metric-card'><div class='m-label'>{lab}</div>"
                        f"<div class='m-value'>{val}</div></div>",
                        unsafe_allow_html=True)

        with c2:
            if 'res' in locals():
                def nano_animation(size, pdi, zeta, drug, oil):
    breakup = min(0.12 + pdi/4, 0.28)
    radius = max(5, min(size/6, 24))

    return f"""
<canvas id="nano"></canvas>
<script>
const c = document.getElementById("nano");
const x = c.getContext("2d");
c.width = 620;
c.height = 340;

let d = [];

class P {{
  constructor(r) {{
    this.x = c.width / 2;
    this.y = c.height / 2;
    this.r = r;
    this.dx = (Math.random() - 0.5) * 1.6;
    this.dy = (Math.random() - 0.5) * 1.6;
  }}

  move() {{
    this.x += this.dx;
    this.y += this.dy;
    if (this.x < 0 || this.x > c.width) this.dx *= -1;
    if (this.y < 0 || this.y > c.height) this.dy *= -1;
  }}

  breakup() {{
    if (this.r > {radius} && Math.random() < {breakup}) {{
      d.push(new P(this.r * 0.65), new P(this.r * 0.65));
      this.r *= 0.55;
    }}
  }}

  draw() {{
    x.beginPath();
    x.arc(this.x, this.y, this.r, 0, Math.PI * 2);
    x.fillStyle = "rgba(0,216,214,0.7)";
    x.shadowBlur = 10;
    x.shadowColor = "#00d8d6";
    x.fill();
  }}
}}

function init() {{
  d = [];
  for (let i = 0; i < 6; i++) {{
    d.push(new P({radius * 2}));
  }}
}}

function animate() {{
  x.fillStyle = "rgba(15,32,39,0.35)";
  x.fillRect(0, 0, c.width, c.height);
  d.forEach(p => {{
    p.move();
    p.breakup();
    p.draw();
  }});
  requestAnimationFrame(animate);
}}

init();
animate();
</script>

<div style="text-align:center;color:#00d8d6;font-size:12px;">
{drug} | {oil} | Size {size:.1f} nm | PDI {pdi:.2f} | Zeta {zeta:.1f} mV
</div>
"""
