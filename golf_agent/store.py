"""Persistent shot history; no LLM-provided database filters or credentials."""
import hashlib
import os

import pandas as pd

from .data import COLUMNS, read_csv, validate


def records(frame):
    df = validate(frame)
    df["Date"] = df.Date.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return df.to_dict(orient="records")


class LocalStore:
    def __init__(self, path):
        self.frame = read_csv(path)

    def list_players(self):
        return sorted(self.frame.Player_ID.unique().tolist())

    def history(self, player_id, limit=100):
        result = self.frame[self.frame.Player_ID == player_id].tail(limit)
        if result.empty:
            raise ValueError("No shots found for that exact player ID")
        return result.copy()


class AstraStore:
    def __init__(self, create=False):
        from astrapy import DataAPIClient
        endpoint = os.environ.get("ASTRA_DB_API_ENDPOINT")
        token = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
        if not endpoint or not token:
            raise ValueError("Set ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN")
        db = DataAPIClient().get_database(endpoint, token=token)
        name = os.environ.get("ASTRA_DB_COLLECTION", "trackman_shots")
        self.collection = db.create_collection(name) if create else db.get_collection(name)

    def list_players(self):
        # Cursor iteration follows all pages; no limit silently truncates the roster.
        # For large clubs, replace this projected scan with a player collection.
        docs = self.collection.find({}, projection={"Player_ID": True, "_id": False})
        return sorted({d["Player_ID"] for d in docs
                       if isinstance(d.get("Player_ID"), str) and d["Player_ID"].strip()})

    def ingest(self, frame):
        # Deterministic IDs + upserts make retries safe after partial ingestion.
        # replace_one(upsert=True) inserts missing documents into the collection.
        completed = 0
        for shot in records(frame):
            key = hashlib.sha256(f"{shot['Player_ID']}|{shot['Date']}".encode()).hexdigest()
            self.collection.replace_one({"_id": key}, {"_id": key, **shot}, upsert=True)
            completed += 1
        return {"upserted": completed}

    def history(self, player_id, limit=100):
        # ISO UTC strings have consistent precision so lexical date sort is temporal.
        docs = list(self.collection.find({"Player_ID": player_id},
                    projection={c: True for c in COLUMNS}, sort={"Date": -1}, limit=limit))
        if not docs:
            raise ValueError("No shots found for that exact player ID")
        return validate(pd.DataFrame(docs))
