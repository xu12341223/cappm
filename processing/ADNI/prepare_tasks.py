"""Align six CAPPM tasks and create subject-disjoint splits from intermediate files."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TASKS = {
    "ANM": ("ANM_diagnosis", "ANMerge_Proteomics_under_90.csv", "mri_meta_2D_3class.pkl", 3),
    "ANM_early": ("ANM_early", "ANMerge_Proteomics_early.csv", "mri_meta_early.pkl", 3),
    "ANM_MMSE": ("ANM_MMSE", "proteomics_MMSE.csv", "mri_meta_2D_MMSE.pkl", 5),
    "ADNI": ("ADNI_diagnosis", "ADNI_Plasma_withlabel.csv", "mri_meta_2D_3class_diagnosislabel.pkl", 3),
    "ADNI_early": ("ADNI_early", "ADNI_Plasma_early.csv", "mri_meta_early.pkl", 3),
    "ADNI_MMSE": ("ADNI_MMSE", "ADNI_Plasma_withMMSE.csv", "mri_meta_2D_MMSE.pkl", 4),
}
LABELS = {"CTL": 0, "NL": 0, "MCI": 1, "AD": 2}
META = {"Unnamed: 0", "Visit", "Month", "Site", "Diagnosis", "Sex", "Age", "APOE", "MMSE",
        "RID", "Visit_Code", "ID", "ID_Visit", "img_array", "label", "MMSE_label"}
SPLITS = ("train", "val", "test")


def subject_ids(ids):
    return pd.Series(ids).astype(str).str.rsplit("_", n=1).str[0]


def load_task(task, raw_root, merge_lowest_mmse=False, duplicate_policy="error", resize_mri=False, adni_mmse_valid_only=False):
    directory, csv_name, pkl_name, classes = TASKS[task]
    source = raw_root / directory
    pp = pd.read_csv(source / csv_name, low_memory=False)
    mri = pd.read_pickle(source / pkl_name)
    excluded_visits = 0
    if task == "ANM":
        # Match the supplied diagnosis preparation script: baseline and month 12 only.
        eligible = pd.to_numeric(pp["Month"], errors="raise").isin([0,12])
        excluded_visits = int((~eligible).sum())
        pp = pp.loc[eligible].reset_index(drop=True)
    if task.startswith("ANM"):
        month = pd.to_numeric(pp["Month"], errors="raise").astype(int)
        ids = pp["Unnamed: 0"].astype(str).str.split("_").str[0] + "_m" + month.astype(str).str.zfill(2)
        pp_labels = pp["MMSE_label"] if task.endswith("MMSE") else pp["Diagnosis"].map(LABELS)
    else:
        ids = pp["RID"].astype(str) + "_" + pp["Visit_Code"].astype(str)
        pp_labels = pp["MMSE_label"] if task.endswith("MMSE") else pp["label"].map(LABELS)
    if "ID_Visit" not in mri:
        rid = mri["RID"].astype(str).str.split("_").str[-1].str.lstrip("0").replace("", "0")
        mri = mri.assign(ID_Visit=rid + "_" + mri["Visit"].astype(str))
    features = [c for c in pp.columns if c not in META and not c.startswith("Unnamed:")]
    x = pp[features].mask(pp[features].eq(".")).apply(pd.to_numeric, errors="raise")
    x = x.replace([np.inf, -np.inf], np.nan)
    x.insert(0, "ID_Visit", ids)
    x["_pp_label"] = pd.to_numeric(pp_labels, errors="raise")
    ycol = "MMSE_label" if task.endswith("MMSE") else "label"
    images = mri[["ID_Visit", "img_array", ycol]].rename(columns={ycol: "_mri_label"}).copy()
    images["_mri_label"] = pd.to_numeric(images["_mri_label"], errors="raise")
    # Keep only sample-aligned records. Conflicting modality labels are counted, not hidden.
    merged = x.merge(images, on="ID_Visit", how="inner")
    matched = len(merged)
    disagreements = int((merged["_pp_label"] != merged["_mri_label"]).sum())
    merged = merged.loc[merged["_pp_label"].eq(merged["_mri_label"])].copy()
    duplicate_rows = int(merged["ID_Visit"].duplicated().sum())
    for _, group in merged.groupby("ID_Visit", sort=False):
        if len(group) > 1:
            if not group[features + ["_pp_label"]].eq(group[features + ["_pp_label"]].iloc[0]).fillna(False).all(axis=None):
                # Treat NaN at the same position as equal.
                a = group[features + ["_pp_label"]].to_numpy(float)
                if not np.allclose(a, a[:1], equal_nan=True):
                    raise ValueError("Conflicting duplicate plasma visit; resolve source data first")
            arrays = [np.asarray(v) for v in group["img_array"]]
            if duplicate_policy == "error" and any(not np.array_equal(arrays[0], v, equal_nan=True) for v in arrays[1:]):
                raise ValueError("Conflicting duplicate MRI visit; resolve source data first")
    merged = merged.drop_duplicates("ID_Visit").reset_index(drop=True)
    merged["label"] = merged["_mri_label"].astype(int)
    if task == "ANM_MMSE" and (merged["label"] == 5).any():
        if not merge_lowest_mmse:
            raise ValueError("ANM_MMSE source has bin 5 but model has 5 classes (0-4). Review labels; explicitly pass --merge-lowest-mmse to map 5 to 4 as in the local ANM preparation script.")
        merged["label"] = merged["label"].clip(upper=4)
    excluded_labels = 0
    if task == "ADNI_MMSE" and adni_mmse_valid_only:
        excluded_labels = int((~merged["label"].isin(range(classes))).sum())
        merged = merged.loc[merged["label"].isin(range(classes))].reset_index(drop=True)
    if not set(merged["label"]).issubset(range(classes)):
        raise ValueError("Labels outside configured class range for " + task)
    if merged.empty:
        raise ValueError("No aligned samples for " + task)
    resized = 0
    if resize_mri:
        converted = []
        for image in merged["img_array"]:
            a = np.asarray(image, dtype=np.float32)
            if a.ndim != 3 or a.shape[0] != 3:
                raise ValueError("MRI must have three channels")
            if a.shape != (3, 72, 72):
                h,w = a.shape[1:]
                yy = np.clip((np.arange(72)+0.5)*h/72-0.5, 0, h-1)
                xx = np.clip((np.arange(72)+0.5)*w/72-0.5, 0, w-1)
                y0,x0 = np.floor(yy).astype(int),np.floor(xx).astype(int)
                y1,x1 = np.minimum(y0+1,h-1),np.minimum(x0+1,w-1)
                wy,wx = (yy-y0)[None,:,None],(xx-x0)[None,None,:]
                a = ((1-wy)*(1-wx)*a[:,y0[:,None],x0[None,:]] + (1-wy)*wx*a[:,y0[:,None],x1[None,:]] + wy*(1-wx)*a[:,y1[:,None],x0[None,:]] + wy*wx*a[:,y1[:,None],x1[None,:]]).astype(np.float32)
                resized += 1
            converted.append(a)
        merged["img_array"] = converted
    for image in merged["img_array"]:
        a = np.asarray(image)
        if a.shape != (3, 72, 72) or not np.isfinite(a).all():
            raise ValueError("MRI must be finite and have shape (3,72,72)")
    return merged, features, {"matched_rows_before_dedup": matched, "modality_label_disagreements_removed": disagreements, "duplicate_visits_removed": duplicate_rows, "mri_resized_to_72": resized, "out_of_scope_labels_removed": excluded_labels, "diagnosis_nonbaseline_non12_rows_excluded": excluded_visits}


def split_subjects(frame, seed):
    # Allocate subjects within dominant-label strata to seven balanced folds.
    # Folds 0/1 are test/validation; 2-6 are training. All visits stay together.
    subjects = subject_ids(frame["ID_Visit"])
    groups = []
    for subject, group in frame.assign(_subject=subjects).groupby("_subject", sort=True):
        counts = group["label"].value_counts()
        label = int(max(counts[counts == counts.max()].index))
        groups.append((subject, label, len(group)))
    rng = np.random.default_rng(seed)
    totals = np.zeros(7, dtype=int)
    assignments = {}
    for label in sorted({g[1] for g in groups}):
        candidates = [g for g in groups if g[1] == label]
        rng.shuffle(candidates)
        candidates.sort(key=lambda g: g[2], reverse=True)
        per_label = np.zeros(7, dtype=int)
        for subject, _, size in candidates:
            choices = np.flatnonzero(per_label == per_label.min())
            fold = int(choices[np.argmin(totals[choices])])
            assignments[subject] = fold
            per_label[fold] += size
            totals[fold] += size
    folds = subjects.map(assignments).to_numpy()
    return {"train": np.flatnonzero(folds >= 2), "val": np.flatnonzero(folds == 1), "test": np.flatnonzero(folds == 0)}


def prepare(task, raw_root, output_root, seed, merge_lowest_mmse, overwrite=False, dry_run=False, duplicate_policy="error", resize_mri=False, adni_mmse_valid_only=False):
    frame, features, summary = load_task(task, raw_root, merge_lowest_mmse, duplicate_policy, resize_mri, adni_mmse_valid_only)
    indices = split_subjects(frame, seed)
    summary.update(task=task, seed=seed, split_level="subject", features=len(features), classes=TASKS[task][3],
                   merge_lowest_mmse=merge_lowest_mmse, duplicate_policy=duplicate_policy, splits={})
    subjects = {}
    output = output_root / task / "overlap"
    if output.exists() and any(output.iterdir()) and not overwrite and not dry_run:
        raise FileExistsError(str(output) + " already has files; use a new output root or --overwrite")
    for split, rows in indices.items():
        part = frame.iloc[rows]
        subjects[split] = set(subject_ids(part["ID_Visit"]))
        if set(part["label"]) != set(range(TASKS[task][3])):
            raise ValueError(task + ": a split is missing a class; inspect subject counts and split feasibility")
        summary["splits"][split] = {"rows": len(part), "subjects": len(subjects[split]), "labels": {str(k): int(v) for k,v in part["label"].value_counts().sort_index().items()}, "visit_order_sha256": hashlib.sha256("\n".join(part["ID_Visit"]).encode()).hexdigest()}
    for a,b in (("train","val"),("train","test"),("val","test")):
        if subjects[a] & subjects[b]:
            raise AssertionError("Subject overlap")
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        for split, rows in indices.items():
            part = frame.iloc[rows].set_index("ID_Visit")
            part[features].to_csv(output / ("X_" + split + "_plasma.csv"))
            part[["img_array"]].to_pickle(output / ("X_" + split + "_img.pkl"))
            part[["label"]].to_csv(output / ("y_" + split + ".csv"))
        (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=[*[k for k in TASKS if k.startswith("ADNI")], "all"], required=True)
    p.add_argument("--raw-root", type=Path, required=True, help="Intermediate input root; see README")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge-lowest-mmse", action="store_true")
    p.add_argument("--duplicate-policy", choices=["error", "first"], default="error", help="Explicit first-scan selection for duplicated MRI visits, preserving source order")
    p.add_argument("--resize-mri", action="store_true", help="Explicit bilinear conversion to (3,72,72); changes original 224px inputs")
    p.add_argument("--adni-mmse-valid-only", action="store_true", help="Explicitly exclude ADNI MMSE bins outside 0-3, following local source script")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    for task in [k for k in TASKS if k.startswith("ADNI")] if args.dataset == "all" else [args.dataset]:
        prepare(task, args.raw_root, args.output_root, args.seed, args.merge_lowest_mmse, args.overwrite, args.dry_run, args.duplicate_policy, args.resize_mri, args.adni_mmse_valid_only)

if __name__ == "__main__":
    main()
