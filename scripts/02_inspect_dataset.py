"""Step 1.2 - look before you build. Finds CT/mask pairs, counts vertebrae, reports which
subjects repeat (patient-level split!) and whether any fracture-grade file exists."""
import argparse, collections, pathlib, re, yaml
from vertfrac.preprocess import VERT_NAMES

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
root = pathlib.Path(yaml.safe_load(open(ap.parse_args().config))["data"]["verse_root"])

cts = sorted(root.rglob("*_ct.nii.gz"))
masks = {re.sub(r"_seg-vert_msk\.nii\.gz$", "", p.name): p for p in root.rglob("*_seg-vert_msk.nii.gz")}
print(f"CT volumes: {len(cts)} | vertebra masks: {len(masks)}")
missing = [c.name for c in cts if re.sub(r"_ct\.nii\.gz$", "", c.name) not in masks]
print(f"CTs without a mask: {len(missing)}", missing[:3])

subs = collections.Counter(re.match(r"(sub-verse\d+)", c.name).group(1) for c in cts if re.match(r"(sub-verse\d+)", c.name))
print(f"unique subject ids: {len(subs)}; ids with >1 scan: {[s for s, n in subs.items() if n > 1][:5]}")

print("\nGrade-like files (json/csv/xlsx/txt mentioning 'grade' or 'fract'):")
hits = [p for p in root.rglob("*") if p.suffix in {".json", ".csv", ".xlsx", ".txt", ".md"}
        and re.search(r"grade|fract|genant", p.name, re.I)]
for p in hits[:20]: print("  ", p)
if not hits:
    print("  none found -> follow ROADMAP step 1.3 (get Loffler et al. grades, Radiology: AI 2020)")
print("\nVertebra labels follow VerSe convention, e.g.", {k: VERT_NAMES[k] for k in (8, 19, 20, 28)})
