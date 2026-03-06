from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'cpt_unimol_project' / 'data'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_task_csv(path: Path, task_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df.rename(columns={'canonical_smiles': 'smiles', 'pIC50': task_col})[['smiles', task_col]]
    df[task_col] = pd.to_numeric(df[task_col], errors='coerce')
    df = df.dropna(subset=['smiles', task_col])
    return df


def main() -> None:
    hep = load_task_csv(ROOT / 'alldata' / 'hepg2_smiles_pIC50.csv', 'hepg2_pIC50')
    hct = load_task_csv(ROOT / 'alldata' / 'hct116_smiles_pIC50.csv', 'hct116_pIC50')

    hep = hep.groupby('smiles', as_index=False)['hepg2_pIC50'].mean()
    hct = hct.groupby('smiles', as_index=False)['hct116_pIC50'].mean()

    master = pd.merge(hep, hct, on='smiles', how='outer')
    master['n_tasks_observed'] = master[['hepg2_pIC50', 'hct116_pIC50']].notna().sum(axis=1)

    out_csv = OUT_DIR / 'master_multitask.csv'
    master.to_csv(out_csv, index=False, encoding='utf-8-sig')

    print('saved', out_csv)
    print('rows_total', len(master))
    print('rows_hepg2', int(master['hepg2_pIC50'].notna().sum()))
    print('rows_hct116', int(master['hct116_pIC50'].notna().sum()))
    print('rows_both_tasks', int((master['n_tasks_observed'] == 2).sum()))


if __name__ == '__main__':
    main()
