import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from golf_agent.agent import ToolRuntime, run_agent
from golf_agent.contracts import TOOLS
from golf_agent.data import generate, validate
from golf_agent.ml import FEATURES, Predictor, features, train
from golf_agent.store import AstraStore, LocalStore


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "shots.csv"
        self.df = generate(self.csv, shots=60)

    def test_rolling_excludes_current_future_and_other_players(self):
        before = features(self.df)
        changed = self.df.copy()
        mask = (changed.Player_ID == "Player A") & (changed.Date >= changed.Date.unique()[10])
        changed.loc[mask, "Spin_Rate"] += 1000
        after = features(changed)
        np.testing.assert_allclose(before.loc[:10, "Spin_Rate_Prior5"],
                                   after.loc[:10, "Spin_Rate_Prior5"], equal_nan=True)
        self.assertTrue(before.groupby("Player_ID").head(1).Spin_Rate_Prior5.isna().all())
        self.assertAlmostEqual(before.loc[10, "Spin_Rate_Prior5"], self.df.loc[5:9, "Spin_Rate"].mean())
        np.testing.assert_allclose(before.loc[60:, "Spin_Rate_Prior5"],
                                   after.loc[60:, "Spin_Rate_Prior5"], equal_nan=True)

    def test_model_round_trip_and_runtime(self):
        report = train(self.df, self.root)
        self.assertEqual(report["train_rows"] + report["test_rows"], len(self.df))
        predictor = Predictor(self.root)
        runtime = ToolRuntime(LocalStore(self.csv), predictor, self.root)
        history = runtime.dispatch("query_player_history", {"player_id": "Player A", "limit": 30})
        prediction = runtime.dispatch("predict_swing_tendency", {"history_id": history["history_id"]})
        self.assertEqual(set(prediction["features"]), set(FEATURES))
        self.assertAlmostEqual(sum(prediction["class_scores"].values()), 1, places=5)
        for kind in ["spin_carry", "delivery_groups"]:
            result = runtime.dispatch("generate_performance_chart", {"history_id": history["history_id"], "kind": kind})
            svg = Path(result["path"]).read_text()
            self.assertIn("<svg", svg)
            self.assertIn("<text", svg)
        with self.assertRaises(Exception):
            runtime.dispatch("query_player_history", {"player_id": "Player A", "limit": 900})
        with self.assertRaises(ValueError):
            runtime.dispatch("predict_swing_tendency", {"history_id": "invented"})

    def test_bad_data(self):
        with self.assertRaises(ValueError):
            validate(self.df.iloc[[0, 0]])
        bad = self.df.copy()
        bad.loc[0, "Spin_Rate"] = np.inf
        with self.assertRaises(ValueError):
            validate(bad)

    def test_astra_idempotent_ingestion_contract(self):
        documents = {}
        class Collection:
            def replace_one(self, filter, replacement, upsert):
                self_test.assertTrue(upsert)
                documents[filter["_id"]] = replacement
        self_test = self
        store = AstraStore.__new__(AstraStore)
        store.collection = Collection()
        store.ingest(self.df)
        store.ingest(self.df)
        self.assertEqual(len(documents), len(self.df))

    def test_agent_continuation_and_final_schema(self):
        brief = {"player_id": "Player A", "summary": "Test", "evidence": [],
                 "swing_tendency": "unavailable", "charts": [], "limitations": [], "next_steps": [],
                 "equipment_assessment": {"can_estimate_shaft_effect": False,
                                          "reason": "No trial data", "suggested_fitting_trial": "Randomized comparison"}}
        call = SimpleNamespace(type="function_call", name="query_player_history", call_id="call_1",
                               arguments=json.dumps({"player_id": "Player A", "limit": 10}))
        responses = [SimpleNamespace(status="completed", output=[call], output_text=""),
                     SimpleNamespace(status="completed", output=[], output_text=json.dumps(brief))]
        requests = []
        def create(**kwargs):
            requests.append({**kwargs, "input": list(kwargs["input"])})
            return responses.pop(0)
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        result = run_agent("Analyze Player A", ToolRuntime(LocalStore(self.csv), None, self.root), "test", client)
        self.assertEqual(result, brief)
        output = requests[1]["input"][-1]
        self.assertEqual(output["call_id"], "call_1")
        self.assertTrue(json.loads(output["output"])["ok"])
        self.assertEqual(requests[0]["tools"], TOOLS)

    def test_agent_budget_and_failed_tool(self):
        call = SimpleNamespace(type="function_call", name="unknown", call_id="bad", arguments="{}")
        response = SimpleNamespace(status="completed", output=[call], output_text="")
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: response))
        with self.assertRaisesRegex(RuntimeError, "budget"):
            run_agent("Analyze Player A", ToolRuntime(None, None, self.root), "test", client, max_rounds=1)


if __name__ == "__main__":
    unittest.main()
