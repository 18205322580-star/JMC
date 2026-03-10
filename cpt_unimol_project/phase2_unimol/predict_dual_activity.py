from pathlib import Path
import os
from typing import Any, Dict, Optional

from unimol_tools import MolPredict


ROOT = Path(__file__).resolve().parents[2]
PHASE2_DIR = ROOT / 'cpt_unimol_project' / 'phase2_unimol'


def _is_valid_model_dir(path: Path) -> bool:
    required = ['config.yaml', 'model_0.pth', 'target_scaler.ss']
    return path.is_dir() and all((path / name).exists() for name in required)


def resolve_model_dir(model_dir: Optional[str] = None) -> Path:
    if model_dir:
        candidate = Path(model_dir)
        if _is_valid_model_dir(candidate):
            return candidate
        raise FileNotFoundError(f'Invalid model directory: {candidate}')

    env_model_dir = os.environ.get('CPT_UNIMOL_MODEL_DIR', '').strip()
    if env_model_dir:
        candidate = Path(env_model_dir)
        if _is_valid_model_dir(candidate):
            return candidate

    preferred = [
        PHASE2_DIR / 'artifacts_unimol_bestparams_e400_probe_20260306',
        PHASE2_DIR / 'artifacts_unimol_tune_r2_gpu_20260306_v1',
        PHASE2_DIR / 'artifacts_unimol_opt_cached_fast_b32_r3_gpu_20260306',
        PHASE2_DIR / 'artifacts_unimol_opt_cached_fast_b128_r3_gpu_20260306',
        PHASE2_DIR / 'artifacts_unimol',
    ]
    for path in preferred:
        if _is_valid_model_dir(path):
            return path

    candidates = [p for p in PHASE2_DIR.glob('artifacts_*') if _is_valid_model_dir(p)]
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    raise FileNotFoundError('No usable model directory found under phase2_unimol.')


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_dual_values(raw_output: Any) -> Dict[str, Optional[float]]:
    # Uni-Mol prediction output may be ndarray/list-like. Normalize to first-row 2-target values.
    row = None
    if hasattr(raw_output, 'tolist'):
        raw_output = raw_output.tolist()

    if isinstance(raw_output, list) and raw_output:
        first = raw_output[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            row = first
        elif len(raw_output) >= 2 and not isinstance(first, (list, tuple)):
            row = raw_output
    elif isinstance(raw_output, tuple) and len(raw_output) >= 2:
        row = list(raw_output)

    if not row:
        return {'hepg2_pIC50': None, 'hct116_pIC50': None}

    return {
        'hepg2_pIC50': _safe_float(row[0]),
        'hct116_pIC50': _safe_float(row[1]),
    }


def predict_one(smiles: str, model_dir: Optional[str] = None) -> Dict[str, Any]:
    smiles = (smiles or '').strip()
    if not smiles:
        raise ValueError('SMILES is empty.')

    resolved_model_dir = resolve_model_dir(model_dir)
    predictor = MolPredict(load_model=str(resolved_model_dir))
    raw_output = predictor.predict({'SMILES': [smiles]})
    values = _extract_dual_values(raw_output)
    return {
        'smiles': smiles,
        'model_dir': str(resolved_model_dir),
        'hepg2_pIC50': values['hepg2_pIC50'],
        'hct116_pIC50': values['hct116_pIC50'],
        'raw': raw_output,
    }


if __name__ == '__main__':
    test_smiles = 'CC1=C2C(=O)OC3(CC)C(=O)OCC3C2=NC4=CC=CC=C14'
    print(predict_one(test_smiles))
