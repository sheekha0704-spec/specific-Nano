# You will need to install: pip install rdkit pubchempy
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp

def get_structure_image(drug_name):
    try:
        # Fetching SMILES from PubChem based on your Drug Name
        results = pcp.get_compounds(drug_name, 'name')
        if results:
            smiles = results[0].canonical_smiles
            mol = Chem.MolFromSmiles(smiles)
            # Create a 2D image of the drug
            img = Draw.MolToImage(mol, size=(300, 300))
            return img, smiles
    except:
        return None, None

# In your Step 1 Page:
st.subheader("Chemical Identity")
img, smiles = get_structure_image(st.session_state.inputs['drug'])
if img:
    st.image(img, caption=f"2D Structure of {st.session_state.inputs['drug']}")
    st.write(f"**SMILES:** `{smiles}`")
