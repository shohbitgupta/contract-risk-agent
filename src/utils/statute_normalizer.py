import re
from typing import Any, Dict, List, Optional


def normalize_act_name(act: Optional[str]) -> str:
    """
    Normalize act naming for consistent downstream matching.
    """
    if not act:
        return "RERA Act, 2016"

    lowered = act.strip().lower()
    if "real estate" in lowered or "rera" in lowered:
        return "RERA Act, 2016"

    return " ".join(act.split())


def normalize_section_ref(ref: str) -> Optional[str]:
    """
    Normalize section references into a canonical form.

    Supported inputs:
    - "Section 18"
    - "section18(1)(a)"
    - "RERA_ACT_SECTION_18_1_A"
    - "18(1)(a)"
    """
    if not ref:
        return None

    token = ref.strip()
    if not token:
        return None

    # Handle anchor format like RERA_ACT_SECTION_18_1_A
    anchor_match = re.match(r"^RERA_ACT_SECTION_(.+)$", token, flags=re.IGNORECASE)
    if anchor_match:
        parts = [p for p in anchor_match.group(1).split("_") if p]
        if not parts:
            return None
        head = parts[0]
        tail = "".join(f"({p.lower()})" for p in parts[1:])
        return f"Section {head}{tail}"

    # Remove optional leading "Section"
    token = re.sub(r"^\s*section\s*", "", token, flags=re.IGNORECASE)
    token = token.strip()
    if not token:
        return None

    # Normalize whitespace and keep only section-like refs.
    token = re.sub(r"\s+", "", token)
    if not re.match(r"^\d+[A-Za-z0-9()]*$", token):
        return None

    return f"Section {token}"


def normalize_state_rule_ref(rule: str) -> Optional[str]:
    if not rule:
        return None
    clean = " ".join(rule.split()).strip()
    return clean or None


def normalize_statutory_basis(
    basis: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Canonicalize statutory basis so anchor matching can be deterministic.
    """
    if not basis:
        return None

    act = normalize_act_name(basis.get("act"))
    sections_raw = basis.get("sections", []) or []
    rules_raw = basis.get("state_rules", []) or []

    normalized_sections: List[str] = []
    for item in sections_raw:
        normalized = normalize_section_ref(str(item))
        if normalized and normalized not in normalized_sections:
            normalized_sections.append(normalized)

    normalized_rules: List[str] = []
    for item in rules_raw:
        normalized = normalize_state_rule_ref(str(item))
        if normalized and normalized not in normalized_rules:
            normalized_rules.append(normalized)

    out: Dict[str, Any] = {"act": act, "sections": normalized_sections}
    if normalized_rules:
        out["state_rules"] = normalized_rules

    return out


def anchors_to_sections(anchors: List[str]) -> List[str]:
    """
    Convert anchor constants to canonical section references.
    Example: RERA_ACT_SECTION_18 -> Section 18
    """
    out: List[str] = []
    for anchor in anchors:
        normalized = normalize_section_ref(anchor)
        if normalized and normalized not in out:
            out.append(normalized)
    return out
