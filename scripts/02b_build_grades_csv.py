"""Step 1.3 - build data/grades.csv (subject,vert_label,grade) from the Genant-grade metadata that ships with
Loffler et al. 2020 (Radiology: AI) - the VerSe'19 OSF project https://osf.io/nqjyw/ (NOT in the s3 zips).
Download that metadata file (xlsx/csv) first, then run e.g.
    python scripts/02b_build_grades_csv.py --file data/verse19_metadata.xlsx
It understands two layouts and PRINTS what it detected so you can check it:
  long : one row per vertebra   (columns: subject id, vertebra level, grade)
  wide : one row per subject    (columns: subject id, then one column per level T1..L5)
If auto-detection is wrong, pass --subject-col/--level-col/--grade-col explicitly.
NOTE: written without access to the real file - verify the printed preview against the paper before training."""
import argparse, re, pathlib, pandas as pd
from vertfrac.preprocess import VERT_NAMES

NAME2LAB = {v: k for k, v in VERT_NAMES.items()}
ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True); ap.add_argument("--sheet", default=0)
ap.add_argument("--subject-col"); ap.add_argument("--level-col"); ap.add_argument("--grade-col")
ap.add_argument("--out", default="data/grades.csv")
a = ap.parse_args()
f = pathlib.Path(a.file)
df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f, sheet_name=a.sheet)
print("columns:", list(df.columns)); print(df.head(8).to_string())


def norm_subject(x):
    m = re.search(r"verse\s*0*(\d+)", str(x), re.I)
    if m: return f"sub-verse{int(m.group(1)):03d}"
    m = re.search(r"(\d+)", str(x))
    return f"sub-verse{int(m.group(1)):03d}" if m else None


def norm_level(x):
    s = str(x).strip().upper()
    if s in NAME2LAB: return NAME2LAB[s]
    try:
        v = int(float(s)); return v if v in VERT_NAMES else None
    except ValueError:
        return None


def norm_grade(x):
    try:
        g = int(float(x)); return g if 0 <= g <= 3 else None
    except (TypeError, ValueError):
        return None


cols = {str(c).strip().upper(): c for c in df.columns}
level_cols = [c for k, c in cols.items() if k in NAME2LAB]
subj = a.subject_col or next((c for k, c in cols.items() if re.search(r"SUBJ|ID|PATIENT|SCAN|NAME", k)), df.columns[0])
rows = []
if len(level_cols) >= 5 and not a.level_col:                       # wide layout
    print(f"layout: WIDE, subject column = {subj!r}, {len(level_cols)} level columns")
    for _, r in df.iterrows():
        for c in level_cols:
            g = norm_grade(r[c])
            if g is not None: rows.append((norm_subject(r[subj]), NAME2LAB[str(c).strip().upper()], g))
else:                                                              # long layout
    lvl = a.level_col or next((c for k, c in cols.items() if re.search(r"LEVEL|VERT|LABEL", k)), None)
    grd = a.grade_col or next((c for k, c in cols.items() if re.search(r"GENANT|GRADE|FRACT", k)), None)
    assert lvl and grd, "Could not detect level/grade columns - pass --level-col and --grade-col"
    print(f"layout: LONG, subject={subj!r}, level={lvl!r}, grade={grd!r}")
    for _, r in df.iterrows():
        lab, g = norm_level(r[lvl]), norm_grade(r[grd])
        if lab is not None and g is not None: rows.append((norm_subject(r[subj]), lab, g))
out = pd.DataFrame(rows, columns=["subject", "vert_label", "grade"]).dropna().drop_duplicates(["subject", "vert_label"])
out = out[out.vert_label.isin(range(8, 26)) | (out.vert_label == 28)]     # thoracolumbar only
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); out.to_csv(a.out, index=False)
print(f"\nwrote {a.out}: {len(out)} graded vertebrae in {out.subject.nunique()} subjects")
print("vertebrae per grade:\n", out.grade.value_counts().sort_index().to_string())
print("\nSanity check vs the paper: VerSe'19 = 160 scans / 141 patients / 1725 vertebrae.")
