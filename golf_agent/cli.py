import argparse
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from .data import generate, read_csv
from .ml import Predictor, train
from .store import AstraStore, LocalStore


def main():
    # Local fitter setup: load credentials without printing them; shell settings win.
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(description="Golf performance prototype")
    parser.add_argument("command", choices=["generate", "train", "ingest", "demo", "chat", "schemas"])
    parser.add_argument("--csv", type=Path, default=Path("data/trackman_mock.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--backend", choices=["local", "astra"], default="astra")
    parser.add_argument("--prompt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "generate":
            print(f"Generated {len(generate(args.csv))} shots: {args.csv}")
        elif args.command == "train":
            print(json.dumps(train(read_csv(args.csv), args.artifacts), indent=2))
        elif args.command == "ingest":
            print(json.dumps(AstraStore(create=True).ingest(read_csv(args.csv))))
        elif args.command == "schemas":
            from .contracts import TOOLS, BRIEF_SCHEMA
            Path("schemas").mkdir(exist_ok=True)
            Path("schemas/tools.json").write_text(json.dumps(TOOLS, indent=2))
            Path("schemas/brief.json").write_text(json.dumps(BRIEF_SCHEMA, indent=2))
        elif args.command == "demo":
            from .charts import chart
            df = generate(args.csv)
            train(df, args.artifacts)
            player = LocalStore(args.csv).history("Player A", 100)
            result = {"mode": "offline tool demonstration, no LLM call",
                      "prediction": Predictor(args.artifacts).predict(player),
                      "charts": [chart(player, kind, args.artifacts) for kind in ["spin_carry", "delivery_groups"]]}
            args.artifacts.joinpath("demo.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        else:
            from .agent import ToolRuntime, run_agent
            store = LocalStore(args.csv) if args.backend == "local" else AstraStore()
            runtime = ToolRuntime(store, Predictor(args.artifacts), args.artifacts)
            prompt = args.prompt or input("Master fitter > ")
            print(json.dumps(run_agent(prompt, runtime, os.environ.get("OPENAI_MODEL", "")), indent=2))
    except Exception as exc:
        # Avoid printing arbitrary SDK exceptions, which can contain connection details.
        logging.error("Command failed (%s). Verify inputs, credentials and artifacts.", type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
