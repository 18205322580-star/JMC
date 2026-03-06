from pathlib import Path
from unimol_tools import MolPredict

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'cpt_unimol_project' / 'phase2_unimol' / 'artifacts_unimol'


def predict_one(smiles: str):
    pred = MolPredict(load_model=str(MODEL_DIR))
    out = pred.predict({'SMILES': [smiles]})
    return out


if __name__ == '__main__':
    test_smiles = 'CC1=C2C(=O)OC3(CC)C(=O)OCC3C2=NC4=CC=CC=C14'
    print(predict_one(test_smiles))
