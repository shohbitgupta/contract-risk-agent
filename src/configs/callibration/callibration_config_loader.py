# src/configs/calibration/calibration_config_loader.py

from pathlib import Path
import yaml
import copy


EXPECTED_ALIGNMENTS = {
    "aligned",
    "partially_aligned",
    "insufficient_evidence",
    "contradiction",
}

EXPECTED_RISKS = {"low", "medium", "high"}


class CalibrationConfig:
    """
    Loads and validates legal calibration thresholds.

    Supports:
    - Central calibration (mandatory)
    - Optional state-specific overrides
    - Strict validation with fail-fast guarantees

    This is a POLICY object, not a heuristic.
    """

    def __init__(
        self,
        central_path: Path,
        state_override_path: Path | None = None,
    ):
        # -------------------------------------------------
        # Load central calibration
        # -------------------------------------------------
        self.central_raw = self._load_yaml(central_path, "central_config")

        # -------------------------------------------------
        # Load optional state override
        # -------------------------------------------------
        self.state_raw = None
        if state_override_path:
            self.state_raw = self._load_yaml(
                state_override_path, "state_overrides"
            )

        # -------------------------------------------------
        # Merge (central + state overrides)
        # -------------------------------------------------
        self.raw = self._merge_calibration(
            self.central_raw,
            self.state_raw.get("overrides") if self.state_raw else None,
        )

        # -------------------------------------------------
        # Mandatory metadata
        # -------------------------------------------------
        self.version = self.raw["version"]
        self.source = self.raw.get("source", "unknown")
        self.sample_size = self.raw.get("sample_size", 0)

        # -------------------------------------------------
        # Core sections
        # -------------------------------------------------
        self.thresholds = self.raw["thresholds"]
        self.weights = self.raw["weights"]
        self.observations = self.raw.get("observations", {})

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------
        self._validate_thresholds()
        self._validate_weights()
        self._validate_observations()

    # =========================================================
    # YAML loading
    # =========================================================

    def _load_yaml(self, path: Path, label: str) -> dict:
        if path is None:
            raise ValueError(f"{label} path must be provided")

        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"{label} file is empty or invalid YAML: {path}")

        return raw

    # =========================================================
    # Deep merge logic
    # =========================================================

    def _merge_calibration(
        self,
        base: dict,
        overrides: dict | None,
    ) -> dict:
        """
        Deep merge with override priority.
        Central calibration is immutable unless explicitly overridden.
        """
        merged = copy.deepcopy(base)

        if not overrides:
            return merged

        def deep_merge(dst: dict, src: dict):
            for key, value in src.items():
                if (
                    key in dst
                    and isinstance(dst[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(dst[key], value)
                else:
                    dst[key] = value

        deep_merge(merged, overrides)
        return merged

    # =========================================================
    # Validation
    # =========================================================

    def _validate_thresholds(self):
        t = self.thresholds

        if not isinstance(t.get("contradiction_fatal"), bool):
            raise ValueError("thresholds.contradiction_fatal must be boolean")

        ratio = t.get("insufficient_evidence_ratio")
        if not isinstance(ratio, (int, float)) or not (0.0 <= ratio <= 1.0):
            raise ValueError(
                "thresholds.insufficient_evidence_ratio must be between 0 and 1"
            )

        score = t.get("high_risk_clause_score")
        if not isinstance(score, (int, float)) or not (0.0 < score < 1.0):
            raise ValueError(
                "thresholds.high_risk_clause_score must be between 0 and 1"
            )

    def _validate_weights(self):
        weights = self.weights

        # Alignment weights
        alignment = weights.get("alignment")
        if not alignment:
            raise ValueError("weights.alignment is required")

        if set(alignment.keys()) != EXPECTED_ALIGNMENTS:
            raise ValueError(
                f"weights.alignment must define exactly {EXPECTED_ALIGNMENTS}"
            )

        # Risk multipliers
        risk = weights.get("risk_multiplier")
        if not risk:
            raise ValueError("weights.risk_multiplier is required")

        if set(risk.keys()) != EXPECTED_RISKS:
            raise ValueError(
                f"weights.risk_multiplier must define exactly {EXPECTED_RISKS}"
            )

    def _validate_observations(self):
        """
        Observations are optional but, if present,
        must be structurally valid.
        """
        for intent, cfg in self.observations.items():
            if not isinstance(cfg, dict):
                raise ValueError(
                    f"Observation for {intent} must be a mapping"
                )

            if "ambiguity_tolerance" in cfg:
                if cfg["ambiguity_tolerance"] not in {
                    "none",
                    "low",
                    "medium",
                    "high",
                }:
                    raise ValueError(
                        f"Invalid ambiguity_tolerance for {intent}"
                    )

    # =========================================================
    # Audit helpers
    # =========================================================

    def audit_metadata(self) -> dict:
        """
        Attach this to ContractAnalysisResult for traceability.
        """
        return {
            "calibration_version": self.version,
            "calibration_source": self.source,
            "sample_size": self.sample_size,
            "state": self.state_raw.get("state")
            if self.state_raw
            else "central",
        }
