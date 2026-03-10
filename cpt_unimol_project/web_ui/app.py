from pathlib import Path
import math
import sys
from typing import Optional

from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpt_unimol_project.phase2_unimol.predict_dual_activity import predict_one, resolve_model_dir


app = Flask(__name__, template_folder='templates')


def _pic50_to_ic50_m(pic50: Optional[float]) -> Optional[float]:
    if pic50 is None:
        return None
    return float(math.pow(10.0, -float(pic50)))


def _build_display_result(raw_result):
    hepg2_pic50 = raw_result.get('hepg2_pIC50')
    hct116_pic50 = raw_result.get('hct116_pIC50')
    hepg2_ic50_m = _pic50_to_ic50_m(hepg2_pic50)
    hct116_ic50_m = _pic50_to_ic50_m(hct116_pic50)
    return {
        **raw_result,
        'hepg2_ic50_m': hepg2_ic50_m,
        'hct116_ic50_m': hct116_ic50_m,
        'hepg2_ic50_nm': (hepg2_ic50_m * 1e9) if hepg2_ic50_m is not None else None,
        'hct116_ic50_nm': (hct116_ic50_m * 1e9) if hct116_ic50_m is not None else None,
    }


@app.route('/', methods=['GET', 'POST'])
def index():
    smiles = ''
    model_dir_input = ''
    default_model = str(resolve_model_dir())
    result = None
    error: Optional[str] = None

    if request.method == 'POST':
        smiles = (request.form.get('smiles') or '').strip()
        model_dir_input = (request.form.get('model_dir') or '').strip()
        try:
            active_model_dir = model_dir_input or None
            raw_result = predict_one(smiles=smiles, model_dir=active_model_dir)
            result = _build_display_result(raw_result)
        except Exception as exc:
            error = str(exc)

    return render_template(
        'index.html',
        smiles=smiles,
        model_dir_input=model_dir_input,
        default_model=default_model,
        result=result,
        error=error,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
