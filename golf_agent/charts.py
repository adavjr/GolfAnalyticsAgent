"""Make: editable SVG evidence for a master fitter's player conversation."""
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .ml import LABELS, features, synthetic_targets


def chart(frame, kind: str, directory: Path) -> dict:
    if kind not in {"spin_carry", "delivery_groups"}:
        raise ValueError("Unsupported chart kind")
    df = features(frame)
    if df.Player_ID.nunique() != 1:
        raise ValueError("Charts expect one player")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{kind}_{uuid4().hex[:12]}.svg"
    # SVG text remains editable in Illustrator; no rasterized scatterplot layer.
    with plt.rc_context({"svg.fonttype": "none"}):
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        try:
            if kind == "spin_carry":
                sns.scatterplot(data=df, x="Spin_Rate", y="Carry_Distance",
                                hue="Club_Speed", palette="viridis", ax=ax)
                ax.set(xlabel="Spin rate (rpm)", ylabel="Carry (yards)",
                       title="Observed spin and carry — association, not an optimum")
            else:
                df["Synthetic tendency"] = [LABELS[i] for i in synthetic_targets(df)]
                sns.scatterplot(data=df, x="Club_Path", y="Face_To_Path",
                                hue="Synthetic tendency", ax=ax)
                ax.axvline(0, color="gray", linewidth=0.8)
                ax.axhline(0, color="gray", linewidth=0.8)
                ax.set(xlabel="Club path (degrees)", ylabel="Face minus path (degrees)",
                       title="Delivery groupings — not measured ball-flight dispersion")
            fig.text(0.02, 0.01, f"{df.Player_ID.iloc[0]} | n={len(df)} | Right-handed driver demo", fontsize=9)
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(target, format="svg")
        finally:
            plt.close(fig)
    return {"path": str(target.resolve()), "kind": kind, "shots": len(df)}
