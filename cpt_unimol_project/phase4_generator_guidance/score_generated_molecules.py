from pathlib import Path
import pandas as pd
from unimol_tools import MolPredict

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'cpt_unimol_project' / 'phase2_unimol' / 'artifacts_unimol'


def rank_file(input_csv: Path, out_csv: Path, w_hep=0.5, w_hct=0.5):
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    if 'SMILES' not in df.columns:
        if 'smiles' in df.columns:
            df = df.rename(columns={'smiles': 'SMILES'})
        elif 'canonical_smiles' in df.columns:
            df = df.rename(columns={'canonical_smiles': 'SMILES'})
        else:
            raise ValueError('input csv must contain SMILES/smiles/canonical_smiles column')

    pred = MolPredict(load_model=str(MODEL_DIR))
    y = pred.predict({'SMILES': df['SMILES'].tolist()})

    # 兼容不同返回结构
    if hasattr(y, 'shape') and len(getattr(y, 'shape', [])) == 2 and y.shape[1] >= 2:
        df['pred_hepg2_pIC50'] = y[:, 0]
        df['pred_hct116_pIC50'] = y[:, 1]
    else:
        raise RuntimeError('unexpected prediction output format from MolPredict')

    df['joint_score'] = w_hep * df['pred_hepg2_pIC50'] + w_hct * df['pred_hct116_pIC50']
    df = df.sort_values('joint_score', ascending=False).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return df


def main():
    inp = ROOT / 'cpt_unimol_project' / 'phase4_generator_guidance' / 'generated_smiles_example.csv'
    out = ROOT / 'cpt_unimol_project' / 'phase4_generator_guidance' / 'generated_ranked.csv'

    if not inp.exists():
        pd.DataFrame({'SMILES': ['CC1=C2C(=O)OC3(CC)C(=O)OCC3C2=NC4=CC=CC=C14']}).to_csv(inp, index=False, encoding='utf-8-sig')

    ranked = rank_file(inp, out)
    print('saved', out)
    print(ranked.head(5))


if __name__ == '__main__':
    main()
