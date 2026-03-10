import csv
import math
from pathlib import Path

BASE = Path(r"d:\kimi2.5program\JMC\alldata")


def to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def norm_unit(u):
    if u is None:
        return None
    return str(u).strip().replace("μ", "u").replace("µ", "u").replace(" ", "").lower()


def value_to_molar(value, unit):
    if value is None or unit is None:
        return None
    factors = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
        "fm": 1e-15,
    }
    u = norm_unit(unit)
    if u not in factors:
        return None
    m = value * factors[u]
    if m <= 0:
        return None
    return m


def molar_to_nm(m):
    if m is None:
        return None
    return m * 1e9


def p_to_molar(p):
    if p is None:
        return None
    return 10 ** (-p)


def round_or_none(v, n=6):
    if v is None:
        return None
    return round(float(v), n)


def calibrate_one(src_csv: Path, out_csv: Path):
    rows = []
    with src_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            std_value = to_float(row.get("standard_value"))
            std_units = row.get("standard_units")
            p_final = to_float(row.get("pchembl_value_final"))

            std_molar = value_to_molar(std_value, std_units)
            std_nm = molar_to_nm(std_molar)

            p_molar = p_to_molar(p_final)
            p_nm = molar_to_nm(p_molar)

            p_from_std = None
            if std_molar is not None and std_molar > 0:
                p_from_std = -math.log10(std_molar)

            abs_delta = None
            if p_from_std is not None and p_final is not None:
                abs_delta = abs(p_final - p_from_std)

            row["standard_value_molar"] = round_or_none(std_molar, 12)
            row["standard_value_nM"] = round_or_none(std_nm, 6)
            row["pchembl_backcalc_molar"] = round_or_none(p_molar, 12)
            row["pchembl_backcalc_nM"] = round_or_none(p_nm, 6)
            row["p_from_standard"] = round_or_none(p_from_std, 6)
            row["p_delta_abs"] = round_or_none(abs_delta, 6)
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def build_ic50_eq_only(rows, out_csv: Path):
    strict = []
    for r in rows:
        st = (r.get("standard_type") or "").strip().upper()
        rel = (r.get("standard_relation") or "").strip()
        nm = to_float(r.get("standard_value_nM"))
        if st == "IC50" and rel == "=" and nm is not None:
            strict.append(r)

    fieldnames = list(strict[0].keys()) if strict else []
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(strict)
    return strict


def summarize(rows, cell_name):
    total = len(rows)
    has_std_nm = sum(1 for r in rows if to_float(r.get("standard_value_nM")) is not None)
    has_back_nm = sum(1 for r in rows if to_float(r.get("pchembl_backcalc_nM")) is not None)
    ic50_eq = sum(
        1 for r in rows
        if (r.get("standard_type") or "").strip().upper() == "IC50"
        and (r.get("standard_relation") or "").strip() == "="
        and to_float(r.get("standard_value_nM")) is not None
    )
    return {
        "cell": cell_name,
        "rows": total,
        "rows_with_standard_nM": has_std_nm,
        "rows_with_backcalc_nM": has_back_nm,
        "rows_ic50_eq_with_nM": ic50_eq,
    }


def main():
    hep_src = BASE / "chembl_hepg2_gnn_ready.csv"
    hct_src = BASE / "chembl_hct116_gnn_ready.csv"

    hep_out = BASE / "chembl_hepg2_gnn_ready_calibrated.csv"
    hct_out = BASE / "chembl_hct116_gnn_ready_calibrated.csv"

    hep_rows = calibrate_one(hep_src, hep_out)
    hct_rows = calibrate_one(hct_src, hct_out)

    combined = hep_rows + hct_rows
    combined_out = BASE / "chembl_hepg2_hct116_gnn_ready_calibrated_combined.csv"
    if combined:
        with combined_out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
            writer.writeheader()
            writer.writerows(combined)

    strict_out = BASE / "chembl_hepg2_hct116_ic50_eq_calibrated.csv"
    strict_rows = build_ic50_eq_only(combined, strict_out)

    summary_path = BASE / "chembl_unit_calibration_summary.csv"
    s1 = summarize(hep_rows, "HepG2")
    s2 = summarize(hct_rows, "HCT116")
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(s1.keys()))
        writer.writeheader()
        writer.writerow(s1)
        writer.writerow(s2)
        writer.writerow({
            "cell": "Combined",
            "rows": len(combined),
            "rows_with_standard_nM": s1["rows_with_standard_nM"] + s2["rows_with_standard_nM"],
            "rows_with_backcalc_nM": s1["rows_with_backcalc_nM"] + s2["rows_with_backcalc_nM"],
            "rows_ic50_eq_with_nM": len(strict_rows),
        })

    print("OUTPUT")
    print(hep_out)
    print(hct_out)
    print(combined_out)
    print(strict_out)
    print(summary_path)
    print("STRICT_IC50_EQ_ROWS", len(strict_rows))


if __name__ == "__main__":
    main()
