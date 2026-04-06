"""Domain Maturity Layer — official status for each domain.

Every domain has a status that determines what claims can be made:
- production-like: validated on real data, can make operational claims
- experimental: works but limited validation, cannot make strong claims
- research-only: research validation only, not for operational decisions
- synthetic-only: tested on synthetic data only, upper bound estimates
- diagnostic: diagnostic tool only, not a prediction system
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DOMAIN_MATURITY: dict[str, dict[str, Any]] = {
    "finance": {
        "status": "production-like",
        "validated": True,
        "n_episodes": 20,
        "precision": 0.78,
        "recall": 0.64,
        "description": "Validated on real data, forward-tested",
    },
    "commodities": {
        "status": "production-like",
        "validated": True,
        "n_episodes": 10,
        "precision": 0.67,
        "recall": 0.67,
        "description": "Validated on real data, high variance",
    },
    "crypto": {
        "status": "production-like",
        "validated": True,
        "n_episodes": 8,
        "precision": None,
        "recall": None,
        "description": "All post-crash, no active bubbles detected",
    },
    "housing": {
        "status": "experimental",
        "validated": False,
        "n_episodes": 10,
        "precision": None,
        "recall": None,
        "description": "Works but limited recall, needs more data/tuning",
    },
    "geology": {
        "status": "research-only",
        "validated": False,
        "n_episodes": None,
        "precision": None,
        "recall": None,
        "description": "Research validation, KS p>0.05, low statistical power",
    },
    "fraud": {
        "status": "synthetic-only",
        "validated": False,
        "n_episodes": None,
        "precision": None,
        "recall": None,
        "description": "Tested on synthetic data only, C-index +27% upper bound",
    },
    "ews": {
        "status": "diagnostic",
        "validated": False,
        "n_episodes": None,
        "precision": None,
        "recall": None,
        "description": "Early warning signals, weak on real data",
    },
    "ai_compute": {
        "status": "research-only",
        "validated": False,
        "n_episodes": None,
        "precision": None,
        "recall": None,
        "description": "Research validation, not for operational decisions",
    },
    "landslide": {
        "status": "synthetic-only",
        "validated": False,
        "n_episodes": None,
        "precision": None,
        "recall": None,
        "description": "Synthetic data, calibrated by Lei & Sornette 2025",
    },
}


@dataclass
class DomainStatus:
    """Status information for a domain."""

    domain: str
    status: str
    validated: bool
    can_claim_validated: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "status": self.status,
            "validated": self.validated,
            "can_claim_validated": self.can_claim_validated,
            "description": self.description,
        }


def get_domain_status(domain: str) -> DomainStatus:
    """Get status information for a domain."""
    info = DOMAIN_MATURITY.get(domain, {
        "status": "unknown",
        "validated": False,
        "n_episodes": None,
        "precision": None,
        "recall": None,
        "description": "Unknown domain",
    })

    return DomainStatus(
        domain=domain,
        status=info["status"],
        validated=info["validated"],
        can_claim_validated=info["validated"],
        description=info["description"],
    )


def can_claim_validated(domain: str) -> bool:
    """Check if domain can make validated claims."""
    return DOMAIN_MATURITY.get(domain, {}).get("validated", False)


def get_status_label(domain: str) -> str:
    """Get status label for a domain."""
    return DOMAIN_MATURITY.get(domain, {}).get("status", "unknown")


def get_all_statuses() -> dict[str, DomainStatus]:
    """Get status for all domains."""
    return {domain: get_domain_status(domain) for domain in DOMAIN_MATURITY}


def format_status_table() -> str:
    """Format all domain statuses as a readable table."""
    lines = ["Domain Maturity Status:", "=" * 60]
    for domain, info in DOMAIN_MATURITY.items():
        icon = {"production-like": "✅", "experimental": "⚠️", "research-only": "🔬", "synthetic-only": "🧪", "diagnostic": "📊"}.get(info["status"], "❓")
        lines.append(f"  {icon} {domain:15s} | {info['status']:20s} | validated={info['validated']}")
    return "\n".join(lines)
