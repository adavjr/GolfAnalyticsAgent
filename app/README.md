# Streamlit dashboard

Install from this folder, then launch from the project root:

```bash
cd app
../.venv/bin/python -m pip install -r requirements.txt
cd ..
.venv/bin/python -m streamlit run app/dashboard.py --server.address localhost --browser.gatherUsageStats false
```

Open http://localhost:8501. The dashboard reads the project-root `.env` without displaying credentials. Shell variables take precedence; restart the server after changing credentials. Astra DB is the default; Local mock CSV is available for demonstrations (analysis still calls OpenAI).

Select a player, enter a question, and click **Analyze player**. Briefs and editable SVG downloads appear below. Each question starts a fresh agent run to limit token usage. Changing player/data source or refreshing the roster clears the draft and prior results. Downloads do not trigger another paid analysis.

The roster scans all pages of the projected Player_ID collection query. Replace this with a player directory collection at scale. Player queries are enforced in Python, not only in the prompt. Request histories and output folders are isolated. Generated charts live in `artifacts/dashboard/`; remove old runs as needed.

This is a local prototype, not a multi-tenant authorization system. Bind to localhost; add authentication and per-fitter player access before publishing it. Existing mock data/label and equipment-effect limitations still apply.

Tests (no live API calls):

```bash
.venv/bin/python -m unittest discover -s app/tests -v
```

Queries are screened locally before analysis using `golf_agent/relevance.py`. Clearly unrelated requests display a retry message without an OpenAI call. The same check protects CLI/backend calls. This keyword/context heuristic supports short player questions but can misclassify unusual phrasing or mixed-topic requests; it is not a semantic classifier or an access-control mechanism.
