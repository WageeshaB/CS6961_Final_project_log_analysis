import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score


def time_aware_split(df, feature_cols, label_cols, test_frac=0.3):
    print("time_aware_split")
    train_parts, test_parts = [], []

    for pid, pdata in df.groupby('patient_id'):
        pdata = pdata.sort_values(['date', 'hour'])
        split = int(len(pdata) * (1 - test_frac))
        train_parts.append(pdata.iloc[:split])
        test_parts.append(pdata.iloc[split:])

    train_df = pd.concat(train_parts)
    test_df = pd.concat(test_parts)

    X_train = train_df[feature_cols].astype(float).values
    y_train = train_df[label_cols].astype(int).values
    X_test = test_df[feature_cols].astype(float).values
    y_test = test_df[label_cols].astype(int).values

    return X_train, X_test, y_train, y_test, train_df, test_df


def ml_random_forest(df, f_cols, l_cols, label_cols):
    # split
    X_train, X_test, y_train, y_test, train_df, test_df = time_aware_split(
        df, f_cols, l_cols
    )

    # scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',  # handles imbalanced labels
            random_state=42,
            n_jobs=-1
        )
    )
    rf.fit(X_train_scaled, y_train)

    # evaluate per label with F1, not accuracy
    y_pred = rf.predict(X_test_scaled)
    for i, label in enumerate(label_cols):
        print(f"\n── {label}")
        print(classification_report(y_test[:, i], y_pred[:, i], digits=3))

    # overall macro F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nOverall macro F1: {macro_f1:.4f}")

    return rf
