import csv
import json
import math
from pathlib import Path

base = Path(r"d:\kimi2.5program\JMC\alldata")


def to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def normalize_unit(unit):
    if unit is None:
        return None
    return str(unit).strip().replace("μ", "u").replace("µ", "u").replace(" ", "").lower()


def value_to_molar(value, unit):
    if value is None or unit is None:
        return None
    u = normalize_unit(unit)
    factors = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
        "fm": 1e-15,
    }
    if u not in factors:
        return None
    m = value * factors[u]
    if m <= 0:
        return None
    return m


def compute_fallback_pchembl(std_type, std_value, std_units):
    allowed = {"ic50", "gi50", "ec50", "ki", "kd", "ac50", "cc50"}
    if std_type is None:
        return None
    if str(std_type).strip().lower() not in allowed:
        return None
    v = to_float(std_value)
    m = value_to_molar(v, std_units)
    if m is None:
        return None
    return -math.log10(m)


def valid_assay_type(v):
    return v in {"F", "B"}


def valid_relation(v):
    return v in {"=", "<", ">"}


def valid_data_comment(comment):
    return comment is None or str(comment).strip() == ""


def cell_match(rec, cell_name):
    cn = cell_name.lower()
    tpn = (rec.get("target_pref_name") or "").lower()
    ad = (rec.get("assay_description") or "").lower()
    return (cn in tpn) or (cn in ad)


OUTPUT_FIELDS = [
    "cell_line",
    "activity_id",
    "assay_chembl_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "target_pref_name",
    "target_organism",
    "assay_type",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value_final",
    "pchembl_source",
    "pchembl_value_raw",
    "data_validity_comment",
]


def process_cell(cell_name, folder):
    rows = []
    stats = {
        "cell": cell_name,
        "total_raw": 0,
        "drop_no_smiles": 0,
        "drop_cell_mismatch": 0,
        "drop_assay_type": 0,
        "drop_relation": 0,
        "drop_validity_comment": 0,
        "drop_no_label": 0,
        "kept": 0,
        "kept_pchembl_direct": 0,
        "kept_pchembl_fallback": 0,
    }

    for fp in sorted(folder.glob("*.json")):
        obj = json.loads(fp.read_text(encoding="utf-8-sig"))
        for rec in obj.get("activities", []):
            stats["total_raw"] += 1

            smiles = rec.get("canonical_smiles")
            if smiles is None or str(smiles).strip() == "":
                stats["drop_no_smiles"] += 1
                continue

            if not cell_match(rec, cell_name):
                stats["drop_cell_mismatch"] += 1
                continue

            assay_type = rec.get("assay_type")
            if not valid_assay_type(assay_type):
                stats["drop_assay_type"] += 1
                continue

            relation = rec.get("standard_relation")
            if not valid_relation(relation):
                stats["drop_relation"] += 1
                continue

            if not valid_data_comment(rec.get("data_validity_comment")):
                stats["drop_validity_comment"] += 1
                continue

            p_raw = to_float(rec.get("pchembl_value"))
            if p_raw is not None:
                p_final = p_raw
                p_source = "pchembl_value"
                stats["kept_pchembl_direct"] += 1
            else:
                p_fb = compute_fallback_pchembl(
                    rec.get("standard_type"),
                    rec.get("standard_value"),
                    rec.get("standard_units"),
                )
                if p_fb is None:
                    stats["drop_no_label"] += 1
                    continue
                p_final = p_fb
                p_source = "fallback_from_standard_value_units"
                stats["kept_pchembl_fallback"] += 1

            row = {
                "cell_line": cell_name,
                "activity_id": rec.get("activity_id"),
                "assay_chembl_id": rec.get("assay_chembl_id"),
                "molecule_chembl_id": rec.get("molecule_chembl_id"),
                "canonical_smiles": smiles,
                "target_pref_name": rec.get("target_pref_name"),
                "target_organism": rec.get("target_organism"),
                "assay_type": assay_type,
                "standard_type": rec.get("standard_type"),
                "standard_relation": relation,
                "standard_value": rec.get("standard_value"),
                "standard_units": rec.get("standard_units"),
                "pchembl_value_final": round(float(p_final), 6),
                "pchembl_source": p_source,
                "pchembl_value_raw": rec.get("pchembl_value"),
                "data_validity_comment": rec.get("data_validity_comment"),
            }
            rows.append(row)
            stats["kept"] += 1

    out_csv = base / f"chembl_{cell_name.lower()}_gnn_ready.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    return stats, rows, out_csv


def main():
    s1, r1, f1 = process_cell("HepG2", base / "chembl_hepg2_pages_10k")
    s2, r2, f2 = process_cell("HCT116", base / "chembl_hct116_pages_10k")

    combined = r1 + r2
    combined_csv = base / "chembl_hepg2_hct116_gnn_ready_combined.csv"
    with combined_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        w.writerows(combined)

    summary_csv = base / "chembl_gnn_ready_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(s1.keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(s1)
        w.writerow(s2)

    print("OUTPUT")
    print(f1)
    print(f2)
    print(combined_csv)
    print(summary_csv)
    print("SUMMARY")
    print(s1)
    print(s2)
    print("combined_kept", len(combined))


if __name__ == "__main__":
    main()
