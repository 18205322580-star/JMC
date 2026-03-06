from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOP1_DIR = ROOT / 'cpt_unimol_project' / 'phase3_transfer' / 'top1_data'
OUT = ROOT / 'cpt_unimol_project' / 'phase3_transfer' / 'top1_transfer_dataset.csv'


def to_molar(v, u):
    if pd.isna(v) or pd.isna(u):
        return np.nan
    try:
        v = float(v)
    except Exception:
        return np.nan
    u = str(u).strip().replace('μ', 'u').replace('µ', 'u').lower()
    factor = {'m': 1.0, 'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9, 'pm': 1e-12}.get(u)
    if factor is None or v <= 0:
        return np.nan
    return v * factor


def main():
    src = TOP1_DIR / 'top1_activities_raw.csv'
    if not src.exists():
        raise FileNotFoundError('run fetch_top1_chembl.py first')

    df = pd.read_csv(src, encoding='utf-8-sig')
    keep = ['canonical_smiles', 'standard_type', 'standard_value', 'standard_units', 'pchembl_value']
    df = df[[c for c in keep if c in df.columns]].copy()

    df = df.rename(columns={'canonical_smiles': 'SMILES'})
    df = df.dropna(subset=['SMILES'])

    if 'pchembl_value' in df.columns:
        df['top1_pActivity'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
    else:
        df['top1_pActivity'] = np.nan

    miss = df['top1_pActivity'].isna()
    molar = [to_molar(v, u) for v, u in zip(df.loc[miss, 'standard_value'], df.loc[miss, 'standard_units'])]
    molar = np.array(molar, dtype=float)
    fill = np.where(np.isfinite(molar) & (molar > 0), -np.log10(molar), np.nan)
    df.loc[miss, 'top1_pActivity'] = fill

    df = df.dropna(subset=['top1_pActivity'])
    df = df.groupby('SMILES', as_index=False)['top1_pActivity'].mean()

    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print('saved', OUT)
    print('rows', len(df))


if __name__ == '__main__':
    main()
