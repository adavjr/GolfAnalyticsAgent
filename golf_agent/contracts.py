"""Strict schemas are shared by API declarations and server-side validation."""

def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def string():
    return {"type": "string"}


TOOLS = [
    {"type": "function", "name": "query_player_history", "strict": True,
     "description": "Retrieve the latest right-handed driver shots from configured storage. Use exact Player_ID such as Player A. Returns a history_id for subsequent tools.",
     "parameters": obj({"player_id": {"type": "string", "minLength": 1, "maxLength": 80},
                        "limit": {"type": "integer", "minimum": 6, "maximum": 200}})},
    {"type": "function", "name": "predict_swing_tendency", "strict": True,
     "description": "Classify the latest observed shot using a retrieved history_id. Python computes model features from trusted stored measurements. Cannot predict shaft effects.",
     "parameters": obj({"history_id": string()})},
    {"type": "function", "name": "generate_performance_chart", "strict": True,
     "description": "Export an editable SVG using a retrieved history_id. Spin/carry is observational; delivery groups are not ball-flight dispersion.",
     "parameters": obj({"history_id": string(),
                        "kind": {"type": "string", "enum": ["spin_carry", "delivery_groups"]}})},
]
BRIEF_SCHEMA = obj({
    "player_id": string(), "summary": string(),
    "evidence": {"type": "array", "items": string()},
    "swing_tendency": string(),
    "equipment_assessment": obj({"can_estimate_shaft_effect": {"type": "boolean", "enum": [False]},
                                 "reason": string(), "suggested_fitting_trial": string()}),
    "charts": {"type": "array", "items": string()},
    "limitations": {"type": "array", "items": string()},
    "next_steps": {"type": "array", "items": string()},
})
