"""Reflect: distinguish a synthetic pattern classifier from validated fitting advice."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from .data import METRICS, validate

LABELS = ["neutral_delivery", "slice_prone_delivery", "hook_prone_delivery"]
ROLLING = ["Club_Path", "Dynamic_Loft", "Spin_Rate"]
FEATURES = METRICS + ["Face_To_Path", "Spin_Loft"] + [f"{c}_Prior5" for c in ROLLING]


def features(frame: pd.DataFrame) -> pd.DataFrame:
    df = validate(frame)
    df["Face_To_Path"] = df.Face_Angle - df.Club_Path
    df["Spin_Loft"] = df.Dynamic_Loft - df.Attack_Angle
    for col in ROLLING:
        # Shift INSIDE each player group: neither current nor future shots enter form.
        # Missing early history stays NaN; XGBoost handles it without future filling.
        df[f"{col}_Prior5"] = df.groupby("Player_ID")[col].transform(
            lambda series: series.shift(1).rolling(5, min_periods=1).mean())
    return df


def synthetic_targets(df: pd.DataFrame) -> np.ndarray:
    """Illustrative thresholds for right-handed drivers; NOT clinical/coaching truth.

    These labels are derived from inputs, so metrics measure rule approximation.
    Replace with independently collected expert labels for actual deployment.
    """
    return np.select([(df.Club_Path < -3) & (df.Face_To_Path > 3),
                      (df.Club_Path > 3) & (df.Face_To_Path < -3)], [1, 2], default=0)


def train(frame: pd.DataFrame, directory: Path) -> dict:
    df = features(frame)
    times = np.sort(df.Date.unique())
    if len(times) < 10:
        raise ValueError("Need at least ten distinct timestamps")
    cutoff = times[int(len(times) * 0.8)]
    earlier = df.Date < cutoff
    y = synthetic_targets(df)
    if set(y[earlier]) != {0, 1, 2}:
        raise ValueError("Training period must contain all three classes")
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08,
                          objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                          tree_method="hist", random_state=42, n_jobs=2)
    model.fit(df.loc[earlier, FEATURES], y[earlier])
    report = classification_report(y[~earlier], model.predict(df.loc[~earlier, FEATURES]),
                                   labels=[0, 1, 2], target_names=LABELS,
                                   output_dict=True, zero_division=0)
    metadata = {"features": FEATURES, "labels": LABELS, "cutoff": str(cutoff),
                "train_rows": int(earlier.sum()), "test_rows": int((~earlier).sum()),
                "evaluation": "chronological, sequential observed-shot classification",
                "synthetic_labels": True, "report": report}
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(directory / "swing_model.json")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


class Predictor:
    def __init__(self, directory: Path):
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata["features"] != FEATURES or self.metadata["labels"] != LABELS:
            raise ValueError("Model feature/label contract does not match this code")
        self.model = XGBClassifier()
        self.model.load_model(directory / "swing_model.json")

    def predict(self, frame: pd.DataFrame) -> dict:
        df = features(frame)
        if df.Player_ID.nunique() != 1:
            raise ValueError("Inference expects one player")
        row = df.iloc[[-1]]
        scores = self.model.predict_proba(row[FEATURES])[0]
        return {"player_id": str(row.Player_ID.iloc[0]), "date": row.Date.iloc[0].isoformat(),
                "tendency": LABELS[int(scores.argmax())],
                "class_scores": dict(zip(LABELS, map(float, scores))),
                "prior_shots_used": min(5, len(df) - 1),
                "features": {k: None if pd.isna(v) else float(v)
                             for k, v in row[FEATURES].iloc[0].items()},
                "limitation": "Synthetic rule labels; scores are uncalibrated, not clinical confidence. No shaft-effect model."}
