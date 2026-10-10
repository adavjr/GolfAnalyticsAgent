"""Observe: give the master fitter consistent, auditable shot measurements."""
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

METRICS = ["Club_Speed", "Attack_Angle", "Club_Path", "Face_Angle",
           "Dynamic_Loft", "Spin_Rate", "Carry_Distance"]
COLUMNS = ["Player_ID", "Date", *METRICS]


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Empty data or missing columns: {sorted(missing)}")
    df = frame[COLUMNS].copy()
    if df.Player_ID.isna().any() or not df.Player_ID.map(
        lambda x: isinstance(x, str) and 0 < len(x.strip()) <= 80
    ).all():
        raise ValueError("Player_ID must be a nonempty string of at most 80 characters")
    df["Date"] = pd.to_datetime(df.Date, utc=True, errors="raise")
    if df.Date.isna().any():
        raise ValueError("Date cannot be missing")
    df[METRICS] = df[METRICS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[METRICS].to_numpy()).all():
        raise ValueError("Shot metrics must be finite")
    if (df[["Club_Speed", "Spin_Rate", "Carry_Distance"]] <= 0).any().any():
        raise ValueError("Speed, spin and carry must be positive")
    if df.duplicated(["Player_ID", "Date"]).any():
        raise ValueError("Use unique shot timestamps per player; ambiguous ordering rejected")
    return df.sort_values(["Player_ID", "Date"]).reset_index(drop=True)


def generate(path: Path, shots: int = 240, seed: int = 42) -> pd.DataFrame:
    if shots < 30:
        raise ValueError("Generate at least 30 shots per player")
    rng = np.random.default_rng(seed)
    rows = []
    for player, bias in [("Player A", -3.5), ("Player B", 0), ("Player C", 3.5)]:
        for i in range(shots):
            speed = rng.normal(100, 7)
            attack = rng.normal(2, 2.5)
            path_angle = rng.normal(bias, 3)
            face = rng.normal(0, 2.5)
            loft = rng.normal(15, 2.5)
            spin = np.clip(2300 + 105 * (loft - attack - 12) + rng.normal(0, 220), 900, 5000)
            carry = max(80, 2.45 * speed - 0.000012 * (spin - 2400) ** 2 + rng.normal(0, 7))
            rows.append([player, pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=i),
                         speed, attack, path_angle, face, loft, spin, carry])
    df = validate(pd.DataFrame(rows, columns=COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def read_csv(path: Path) -> pd.DataFrame:
    return validate(pd.read_csv(path))


def fetch_csv(url: str, destination: Path) -> pd.DataFrame:
    """Operator-only adapter for an authorized HTTPS CSV export, not a TrackMan API.

    Never expose arbitrary URLs as an LLM tool. Add a host allowlist in deployment.
    """
    if not url.startswith("https://"):
        raise ValueError("An HTTPS export URL is required")
    with requests.Session() as session:
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 502, 503, 504],
                      allowed_methods=["GET"])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        with session.get(url, timeout=(5, 30), stream=True, allow_redirects=False) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Expected HTTP 200; redirects are not followed")
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > 10_000_000:
                    raise ValueError("CSV export exceeds 10 MB")
    df = validate(pd.read_csv(StringIO(content.decode("utf-8-sig"))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return df
