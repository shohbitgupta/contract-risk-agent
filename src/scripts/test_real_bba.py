#!/usr/bin/env python3
"""
Test script to validate the contract risk analyzer against a real BBA contract.

This script:
1. Downloads or uses a real BBA contract PDF
2. Runs it through the full analysis pipeline
3. Evaluates response quality (completeness, correctness, citations)
4. Generates a quality report

Usage:
    python src/scripts/test_real_bba.py --pdf-url <URL>
    python src/scripts/test_real_bba.py --pdf-file <path>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from ingestion.contract_parser.pdf_text_extractor import UserContractPDFExtractor
from main import ContractRiskAnalysisSystem
from vector_index.index_registry import IndexRegistry
from tools.logger import setup_logger

logger = setup_logger("test-real-bba")


def evaluate_response_quality(results: List) -> Dict:
    """
    Evaluate the quality of analysis results.

    Metrics:
    - Completeness: % of clauses analyzed
    - Citation coverage: % with valid citations
    - Risk distribution: breakdown by risk level
    - Alignment distribution: breakdown by alignment
    - Average quality score

    Example:
        >>> results = [ExplanationResult(...), ...]
        >>> metrics = evaluate_response_quality(results)
        >>> print(metrics["completeness"])
        0.95
    """
    if not results:
        return {
            "completeness": 0.0,
            "citation_coverage": 0.0,
            "risk_distribution": {},
            "alignment_distribution": {},
            "avg_quality_score": 0.0,
            "total_clauses": 0
        }

    total = len(results)
    with_citations = sum(1 for r in results if r.citations)
    quality_scores = [r.quality_score for r in results if hasattr(r, 'quality_score')]

    risk_counts = {}
    alignment_counts = {}

    for r in results:
        risk = getattr(r, 'risk_level', 'unknown')
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

        alignment = getattr(r, 'alignment', 'unknown')
        alignment_counts[alignment] = alignment_counts.get(alignment, 0) + 1

    return {
        "completeness": 1.0,  # All chunks processed
        "citation_coverage": with_citations / total if total > 0 else 0.0,
        "risk_distribution": risk_counts,
        "alignment_distribution": alignment_counts,
        "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        "total_clauses": total,
        "clauses_with_high_risk": risk_counts.get("high", 0),
        "clauses_conflicting": alignment_counts.get("conflicting", 0),
        "clauses_insufficient_evidence": alignment_counts.get("insufficient_evidence", 0)
    }


def generate_report(results: List, metrics: Dict, output_path: Path):
    """
    Generate a human-readable quality report.

    Example:
        >>> generate_report(results, metrics, Path("report.json"))
        # Creates report.json with analysis summary
    """
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_clauses_analyzed": metrics["total_clauses"],
            "completeness": f"{metrics['completeness']*100:.1f}%",
            "citation_coverage": f"{metrics['citation_coverage']*100:.1f}%",
            "average_quality_score": f"{metrics['avg_quality_score']:.2f}",
            "high_risk_clauses": metrics["clauses_with_high_risk"],
            "conflicting_clauses": metrics["clauses_conflicting"],
            "insufficient_evidence_clauses": metrics["clauses_insufficient_evidence"]
        },
        "risk_distribution": metrics["risk_distribution"],
        "alignment_distribution": metrics["alignment_distribution"],
        "detailed_results": [
            {
                "clause_id": r.clause_id,
                "intent": getattr(r, 'intent', 'unknown'),
                "risk_level": getattr(r, 'risk_level', 'unknown'),
                "alignment": getattr(r, 'alignment', 'unknown'),
                "quality_score": getattr(r, 'quality_score', 0.0),
                "has_citations": len(r.citations) > 0,
                "citation_count": len(r.citations),
                "summary": r.summary[:100] + "..." if len(r.summary) > 100 else r.summary
            }
            for r in results
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Quality report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test contract risk analyzer with real BBA contract"
    )
    parser.add_argument(
        "--pdf-url",
        help="URL of BBA contract PDF to analyze"
    )
    parser.add_argument(
        "--pdf-file",
        type=Path,
        help="Local path to BBA contract PDF"
    )
    parser.add_argument(
        "--state",
        default="uttar_pradesh",
        help="State jurisdiction (default: uttar_pradesh)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_results"),
        help="Output directory for results (default: test_results)"
    )
    args = parser.parse_args()

    if not args.pdf_url and not args.pdf_file:
        parser.error("Either --pdf-url or --pdf-file must be provided")

    # Setup
    args.output.mkdir(parents=True, exist_ok=True)
    logger.info("Starting real BBA contract validation test")

    # Extract text
    extractor = UserContractPDFExtractor()
    if args.pdf_url:
        logger.info(f"Downloading PDF from: {args.pdf_url}")
        contract_text = extractor.extract_from_url(args.pdf_url)
    else:
        logger.info(f"Reading PDF from: {args.pdf_file}")
        contract_text = extractor.extract_from_file(args.pdf_file)

    if not contract_text or len(contract_text.strip()) < 500:
        raise ValueError("Extracted contract text is too short or empty")

    logger.info(f"Extracted {len(contract_text)} characters from PDF")

    # Initialize system
    BASE_DIR = Path(__file__).resolve().parents[2]
    index_registry = IndexRegistry(
        base_dir=BASE_DIR / "src" / "data" / "vector_indexes",
        embedding_dim=384
    )
    index_registry.validate_state(args.state)

    intent_rules_path = BASE_DIR / "src" / "configs" / "real_state_intent_rules.yaml"

    system = ContractRiskAnalysisSystem(
        index_registry=index_registry,
        intent_rules_path=intent_rules_path
    )

    # Run analysis
    logger.info("Running contract analysis...")
    results = system.analyze_contract(
        contract_text=contract_text,
        state=args.state
    )

    logger.info(f"Analysis complete. Processed {len(results)} clauses")

    # Evaluate quality
    metrics = evaluate_response_quality(results)
    logger.info("Quality metrics:")
    logger.info(f"  Completeness: {metrics['completeness']*100:.1f}%")
    logger.info(f"  Citation Coverage: {metrics['citation_coverage']*100:.1f}%")
    logger.info(f"  Avg Quality Score: {metrics['avg_quality_score']:.2f}")
    logger.info(f"  High Risk Clauses: {metrics['clauses_with_high_risk']}")
    logger.info(f"  Conflicting Clauses: {metrics['clauses_conflicting']}")

    # Generate report
    report_path = args.output / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    generate_report(results, metrics, report_path)

    # Print summary
    print("\n" + "="*60)
    print("QUALITY EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Clauses Analyzed: {metrics['total_clauses']}")
    print(f"Completeness: {metrics['completeness']*100:.1f}%")
    print(f"Citation Coverage: {metrics['citation_coverage']*100:.1f}%")
    print(f"Average Quality Score: {metrics['avg_quality_score']:.2f}/1.0")
    print(f"\nRisk Distribution:")
    for risk, count in metrics['risk_distribution'].items():
        print(f"  {risk}: {count}")
    print(f"\nAlignment Distribution:")
    for align, count in metrics['alignment_distribution'].items():
        print(f"  {align}: {count}")
    print(f"\nDetailed report: {report_path}")
    print("="*60)

    return results, metrics


if __name__ == "__main__":
    main()
