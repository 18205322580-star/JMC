from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / '.venv' / 'Scripts' / 'python.exe'

scripts = [
    ROOT / 'cpt_unimol_project' / 'phase1_3d' / 'build_master_table.py',
    ROOT / 'cpt_unimol_project' / 'phase1_3d' / 'generate_conformers.py',
]

for s in scripts:
    print('RUN', s)
    subprocess.check_call([str(PY), str(s)], cwd=str(ROOT))

print('phase1 done')
