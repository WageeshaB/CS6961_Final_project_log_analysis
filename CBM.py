import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
import statistics
from datetime import datetime


def represent_patients_as_single_row(df, feature_cols):
    result = {}
    for pid, pdata in df.groupby("patient_id"):
        result[pid] = pdata[feature_cols].astype(float).mean().values

    return pd.DataFrame(result, index=feature_cols)


def get_similarity_df(patient_repr, similarity_measurement=2, ):
    if similarity_measurement == 1:  # Pearson correlation
        sim_df = patient_repr.corr().fillna(-1)
        sim_df = (sim_df + 1) / 2.0

    elif similarity_measurement == 2:  # Cosine similarity
        patient_ids = list(patient_repr.columns)
        cos_arr = cosine_similarity(patient_repr.T)
        sim_df = pd.DataFrame(cos_arr, index=patient_ids, columns=patient_ids)

    # normalize to [0, 1]
    mn, mx = sim_df.min().min(), sim_df.max().max()
    if mx > mn:
        sim_df = (sim_df - mn) / (mx - mn)

    return sim_df


def get_community_df(df, patient_id, sim_df, threshold) -> tuple[pd.DataFrame, int]:
    sim_row = sim_df.loc[patient_id].drop(labels=patient_id)
    # print(sim_row)
    similar_patients = sim_row[sim_row >= threshold].index.tolist()
    community_size = len(similar_patients)

    print(f"  Patients within threshold {threshold}: {community_size}")

    if community_size == 0:
        return pd.DataFrame(), 0

    community_df = df[df["patient_id"].isin(similar_patients)].copy()
    return community_df, community_size


def single_iteration(community_df, patient_df, feature_cols, label_cols):
    comm_sample = community_df.sample(frac=0.95, replace=False)

    patient_df_sorted = patient_df.sort_values(["date", "hour"])
    split_idx = int(len(patient_df_sorted) * 0.7)
    train_pat = patient_df_sorted.iloc[:split_idx]
    test_pat = patient_df_sorted.iloc[split_idx:]

    # train_pat, test_pat = train_test_split(patient_df, test_size=0.7)

    train_df = pd.concat([comm_sample, train_pat], ignore_index=True)

    for lc in label_cols: # each label has 2 classes
        if train_df[lc].nunique() < 2:
            return None

    X_train = train_df[feature_cols].astype(float).values
    y_train = train_df[label_cols].astype(int).values
    X_test = test_pat[feature_cols].astype(float).values
    y_test = test_pat[label_cols].astype(int).values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"    y_train: {y_train.shape}, y_test: {y_test.shape}")

    # print(y_train.value_counts())
    # print(y_test.value_counts())

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=None,  # intentionally random across iterations
            n_jobs=-1,
        )
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # return f1_score(y_test, y_pred, average="macro", zero_division=0)
    return {
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }



def get_stats(arr):
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan,
                "min": np.nan, "max": np.nan}
    if len(arr) == 1:
        v = arr[0]
        return {"mean": v, "std": 0.0, "median": v, "min": v, "max": v}
    return {
        "mean":   statistics.mean(arr),
        "std":    statistics.stdev(arr),
        "median": statistics.median(arr),
        "min":    min(arr),
        "max":    max(arr),
    }


def run_iterations(community_df, patient_df, feature_cols, label_cols, n_iterations=5):
    scores = {"f1": [], "precision": [], "recall": []}
    attempts = 0
    max_attempts = n_iterations * 3

    while len(scores["f1"]) < n_iterations and attempts < max_attempts:
        attempts += 1
        result = single_iteration(community_df, patient_df, feature_cols, label_cols)
        if result is not None:
            print(f"      iteration {len(scores['f1']) + 1} → "
                  f"f1={result['f1']:.4f}  "
                  f"precision={result['precision']:.4f}  "
                  f"recall={result['recall']:.4f}")
            for metric in scores:
                scores[metric].append(result[metric])

    if not scores["f1"]:
        return None

    return {
        "f1": get_stats(scores["f1"]),
        "precision": get_stats(scores["precision"]),
        "recall": get_stats(scores["recall"]),
        "raw": scores,
    }


def community_based_rf(df, feature_cols, label_cols, similarity_measurement=2, thresholds=None, n_iterations=5,
                       min_patient_rows=30):
    if thresholds is None:
        thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]

    # normalize
    scaler_global = MinMaxScaler()
    df_norm = df.copy()
    df_norm[feature_cols] = scaler_global.fit_transform(
        df[feature_cols].astype(float)
    )

    # patient 1 row representation & similarity matrix
    patient_repr = represent_patients_as_single_row(df_norm, feature_cols)
    sim_df = get_similarity_df(patient_repr, similarity_measurement)

    print(patient_repr)
    print(sim_df)

    # filter patients
    all_patients = df["patient_id"].unique()
    eligible = []
    for pid in all_patients:
        pdata = df[df["patient_id"] == pid]
        if len(pdata) < min_patient_rows:
            continue
        # each label has at least 2 classes
        if any(pdata[lc].nunique() < 2 for lc in label_cols):
            continue
        eligible.append(pid)

    print(f"Eligible patients: {len(eligible)} / {len(all_patients)}\n")

    all_results = []

    METRICS = ["f1", "precision", "recall"]
    STATS = ["mean", "std", "median", "min", "max"]
    nan_row = {f"{m}_{s}": np.nan for m in METRICS for s in STATS}

    for idx, pid in enumerate(eligible, 1):
        t_patient_start = datetime.now()
        patient_df = df[df["patient_id"] == pid].sort_values(["date", "hour"])

        print(f"\n[{idx}/{len(eligible)}] Patient: {pid}  |  rows: {len(patient_df)}")

        for th in thresholds:
            community_df, com_size = get_community_df(df_norm, pid, sim_df, threshold=th)

            row_base = {
                "patient_id": pid,
                "threshold": th,
                "community_size": com_size,
            }

            if com_size == 0:
                print(f"    th={th} → no community")
                all_results.append({**row_base, **nan_row})
                continue

            combined = pd.concat([community_df, patient_df], ignore_index=True)
            label_ok = all(combined[lc].nunique() >= 2 for lc in label_cols)
            if not label_ok:
                print(f"    th={th} → not all labels in df")
                all_results.append({**row_base, **nan_row})
                continue

            stats = run_iterations(
                community_df=community_df,
                patient_df=patient_df,
                feature_cols=feature_cols,
                label_cols=label_cols,
                n_iterations=n_iterations,
            )

            if stats is None:
                print(f"    th={th} → not enough data")
                all_results.append({**row_base, **nan_row})
                continue

            flat = {f"{m}_{s}": stats[m][s] for m in METRICS for s in STATS}
            print(flat)

            print(
                f"    th={th:.2f}  |  com={com_size:3d}  "
                f"|  F1={flat['f1_mean']:.4f}±{flat['f1_std']:.4f}  "
                f"Precision={flat['precision_mean']:.4f}±{flat['precision_std']:.4f}  "
                f"Recall={flat['recall_mean']:.4f}±{flat['recall_std']:.4f}"
            )
            all_results.append({**row_base, **flat})

        print(f"  Patient {pid} done in {datetime.now() - t_patient_start}")

        # break

    results_df = pd.DataFrame(all_results)


    print("\n" + "================================")
    print("SUMMARY")
    print("================================")
    summary = (
        results_df.dropna(subset=["f1_mean"])
        .groupby("threshold")[["f1_mean", "precision_mean", "recall_mean"]]
        .agg(["mean", "std", "count"])
    )
    print(summary.to_string())

    return results_df
