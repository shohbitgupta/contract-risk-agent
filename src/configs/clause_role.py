from typing import Literal, Set

ClauseRole = Literal[
    "obligation",
    "right",
    "procedure",
    "definition",
    "schedule",
    "boilerplate",
    "reference",
]

IMPACTFUL_ROLES: Set[str] = {
    "obligation",
    "right",
    "procedure",
}

ROLE_WEIGHTS = {
    "obligation": 1.0,
    "right": 1.0,
    "procedure": 0.7,
    "definition": 0.0,
    "schedule": 0.0,
    "boilerplate": 0.2,
    "reference": 0.0,
}