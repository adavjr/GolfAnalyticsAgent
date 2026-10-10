"""Local, conservative topic screening; no API calls or paid classification."""
import re

MESSAGE = "This query does not relate to the available golf performance data. Please try again."


class UnrelatedQueryError(ValueError):
    """The query has no recognizable connection to the available analysis."""


def validate_relevance(prompt: str) -> None:
    # Support golf terms and contextual follow-ups without requiring 'golf'.
    # This is a usability filter, not semantic classification or a security boundary.
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    golf = {
        "golf", "golfer", "golfers", "trackman", "swing", "swings", "shot", "shots",
        "driver", "drivers", "club", "clubs", "shaft", "shafts", "loft", "spin",
        "carry", "slice", "slicing", "hook", "hooking", "fade", "draw", "ball",
        "launch", "attack", "face", "path", "fitting", "fitter", "fairway", "iron",
        "irons", "wedge", "putting", "backswing", "downswing", "dispersion",
    }
    analysis = {
        "analyze", "analyse", "analysis", "summarize", "summarise", "summary",
        "compare", "comparison", "trend", "trends", "consistent", "consistency",
        "average", "averages", "metrics", "measurements", "tendency", "tendencies",
        "chart", "charts", "plot", "graph", "history", "performance", "speed",
        "distance", "recent", "latest", "improve", "improvement",
    }
    context = {"player", "players", "they", "their", "them", "my", "me", "these", "those", "data", "results"}
    if words & golf or (words & analysis and words & context):
        return
    if words and words <= analysis | {"please", "show", "a", "the", "generate", "create", "and", "of", "an"}:
        return
    raise UnrelatedQueryError(MESSAGE)
