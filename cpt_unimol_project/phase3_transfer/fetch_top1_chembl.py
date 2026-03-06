from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'cpt_unimol_project' / 'phase3_transfer' / 'top1_data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = 'https://www.ebi.ac.uk/chembl/api/data'


def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_target_ids(keyword='topoisomerase 1'):
    data = get_json(f'{BASE}/target', params={'pref_name__icontains': keyword, 'format': 'json', 'limit': 200})
    return data.get('targets', [])


def fetch_activities(target_chembl_id):
    all_rows = []
    offset = 0
    limit = 1000
    while True:
        params = {
            'target_chembl_id': target_chembl_id,
            'standard_type__in': 'IC50,Ki,Kd,EC50',
            'standard_relation': '=',
            'format': 'json',
            'limit': limit,
            'offset': offset,
        }
        data = get_json(f'{BASE}/activity', params=params)
        rows = data.get('activities', [])
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        offset += limit
    return all_rows


def main():
    targets = fetch_target_ids()
    tdf = pd.DataFrame(targets)
    tdf.to_csv(OUT_DIR / 'top1_targets.csv', index=False, encoding='utf-8-sig')

    if tdf.empty:
        print('no_targets_found')
        return

    # 优先 human Top1
    pick = tdf[tdf.get('organism', '').eq('Homo sapiens')] if 'organism' in tdf.columns else tdf
    target_id = (pick.iloc[0]['target_chembl_id'] if not pick.empty else tdf.iloc[0]['target_chembl_id'])

    acts = fetch_activities(target_id)
    adf = pd.DataFrame(acts)
    adf.to_csv(OUT_DIR / 'top1_activities_raw.csv', index=False, encoding='utf-8-sig')

    print('target_id', target_id)
    print('n_targets', len(tdf), 'n_activities', len(adf))
    print('saved', OUT_DIR)


if __name__ == '__main__':
    main()
