"""Master fitter dashboard. Run from the project root with Streamlit."""
import json
import logging
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import streamlit as st
from golf_agent.agent import ToolRuntime, run_agent
from golf_agent.relevance import validate_relevance, UnrelatedQueryError
from golf_agent.ml import Predictor
from golf_agent.store import AstraStore, LocalStore

st.set_page_config(page_title="Golf Performance", page_icon="⛳", layout="wide")


def clear_analysis():
    # Changing player clears both the draft and the prior player's visible evidence.
    for key in ("brief", "query", "error", "chart_files", "last_query"):
        st.session_state.pop(key, None)


def begin_analysis():
    # Form callbacks run before the next render, so the button is disabled
    # before the blocking agent call begins (not after it finishes).
    if not st.session_state.get("running", False):
        st.session_state.pending_query = st.session_state.get("query", "")
        st.session_state.running = bool(st.session_state.pending_query.strip())


def get_store(backend):
    return AstraStore() if backend == "Astra DB" else LocalStore(ROOT / "data/trackman_mock.csv")


def render_brief(brief):
    st.subheader("Performance brief")
    st.write(brief["summary"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Swing tendency")
        st.write(brief["swing_tendency"])
        st.markdown("#### Evidence")
        for item in brief["evidence"]:
            st.write("• " + item)
    with right:
        st.markdown("#### Equipment assessment")
        st.write(brief["equipment_assessment"]["reason"])
        st.markdown("#### Suggested fitting trial")
        st.write(brief["equipment_assessment"]["suggested_fitting_trial"])
    st.markdown("#### Charts")
    files = st.session_state.get("chart_files", {})
    if not files:
        st.caption("No chart generated. Ask for a spin/carry chart or delivery groupings.")
    for path, svg in files.items():
        st.image(svg.decode("utf-8"))
        st.download_button("Download editable SVG", svg, file_name=Path(path).name,
                           mime="image/svg+xml", key=path)
    with st.expander("Limitations and next steps", expanded=True):
        for item in brief["limitations"] + brief["next_steps"]:
            st.write("• " + item)
    st.download_button("Download brief JSON", json.dumps(brief, indent=2),
                       file_name="performance_brief.json", mime="application/json")


st.title("⛳ Golf Performance")
st.caption("Player history, delivery insights, and evidence for your next fitting conversation.")
running = st.session_state.get("running", False)
with st.sidebar:
    st.header("Fitting session")
    backend = st.selectbox("Data source", ["Astra DB", "Local mock CSV"],
                           key="backend", on_change=clear_analysis, disabled=running)
    refresh = st.button("Refresh players", disabled=running)
    st.caption("Synthetic right-handed driver demo. Equipment effects require controlled trials.")

# Keep roster within this browser session; refreshing intentionally re-queries storage.
try:
    if refresh or st.session_state.get("roster_backend") != backend:
        clear_analysis()
        with st.spinner("Loading players…"):
            players = get_store(backend).list_players()
        st.session_state.roster = players
        st.session_state.roster_backend = backend
    players = st.session_state.roster
except Exception as exc:
    logging.warning("Roster load failed (%s)", type(exc).__name__)
    st.error("Could not load players. Check the data source and Astra settings in the project .env, then refresh.")
    st.stop()
if not players:
    st.info("No players found. Ingest shot data, then refresh the player list.")
    st.stop()
if st.session_state.get("player") not in players:
    st.session_state.player = players[0]
    clear_analysis()
player = st.sidebar.selectbox("Player", players, key="player", on_change=clear_analysis, disabled=running)
st.subheader(f"Analyzing: {player}")
st.caption("Each question is an independent analysis of this player. Change the selection to analyze someone else.")
ready = bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"))
if not ready:
    st.warning("Set OPENAI_API_KEY and OPENAI_MODEL in the project .env before running an analysis.")
with st.form("question"):
    prompt = st.text_area("Ask the performance agent", key="query", height=110, disabled=running,
                          placeholder="Analyze recent driver delivery and generate a spin/carry chart.")
    submitted = st.form_submit_button("Analyze player", disabled=not ready or running, type="primary",
                                      on_click=begin_analysis)
if running:
    prompt = st.session_state.pending_query
if submitted or running:
    st.session_state.pop("brief", None)
    st.session_state.pop("error", None)
    st.session_state.pop("chart_files", None)
    if not prompt.strip():
        st.warning("Enter a question first.")
    else:
        try:
            validate_relevance(prompt)
            # Fresh runtime and unique output directory isolate histories across requests.
            output = ROOT / "artifacts" / "dashboard" / uuid4().hex
            with st.spinner("Retrieving history, analyzing delivery, and preparing your brief…"):
                runtime = ToolRuntime(get_store(backend), Predictor(ROOT / "artifacts"),
                                      output, selected_player=player)
                brief = run_agent(prompt, runtime, os.environ["OPENAI_MODEL"])
                files = {}
                for raw in brief["charts"]:
                    path = Path(raw).resolve()
                    # Never open an arbitrary file path supplied by an LLM.
                    if path.parent != output.resolve() or path.suffix != ".svg" or not path.is_file():
                        raise ValueError("Invalid chart artifact")
                    files[str(path)] = path.read_bytes()
                st.session_state.brief = brief
                st.session_state.chart_files = files
        except UnrelatedQueryError as exc:
            st.session_state.error = str(exc)
        except Exception as exc:
            logging.warning("Dashboard analysis failed (%s)", type(exc).__name__)
            st.session_state.error = (
                "Analysis could not finish. Check API credits/model access, Astra connectivity, "
                "and trained model artifacts, then retry. No incomplete brief is displayed.")
        finally:
            st.session_state.running = False
        # Render the enabled controls again after either success or failure.
        st.rerun()
if st.session_state.get("error"):
    st.error(st.session_state.error)
if st.session_state.get("brief"):
    render_brief(st.session_state.brief)
