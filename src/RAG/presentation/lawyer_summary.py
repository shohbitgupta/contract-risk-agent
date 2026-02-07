from typing import List, Literal
from pydantic import BaseModel


class LawyerFriendlySummary(BaseModel):
    verdict: Literal[
        "safe_to_sign",
        "review_required",
        "do_not_sign"
    ]

    headline: str
    why_this_matters: List[str]
    key_risk_statistics: List[str]
    critical_clauses: List[str]
    recommended_next_steps: List[str]
