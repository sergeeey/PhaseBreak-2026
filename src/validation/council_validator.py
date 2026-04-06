"""Adversarial AI Council for LPPLS prediction validation.

Architecture:
  LPPLS fit → tc prediction → Council
                                ├── Bull Agent  ("tc is correct, crash imminent")
                                ├── Bear Agent  ("tc is wrong, noise pattern")
                                └── Skeptic Agent ("check methodology")
                                        ↓
                                  Arbiter Agent (weighted synthesis → final verdict)

Uses Ollama local models for zero-cost inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import structlog

from src.lppls.model import LPPLSParams

log = structlog.get_logger()

OLLAMA_URL = "http://localhost:11434/api/generate"


class AgentRole(Enum):
    BULL = "bull"
    BEAR = "bear"
    SKEPTIC = "skeptic"
    ARBITER = "arbiter"


@dataclass
class AgentOpinion:
    """Single agent's assessment of an LPPLS prediction."""

    role: AgentRole
    verdict: str  # BUBBLE / NO_BUBBLE / UNCERTAIN
    confidence: float  # 0-1
    reasoning: str
    key_arguments: list[str] = field(default_factory=list)


@dataclass
class CouncilVerdict:
    """Final council verdict after adversarial debate."""

    final_verdict: str  # BUBBLE / NO_BUBBLE / UNCERTAIN
    final_confidence: float  # 0-1
    opinions: list[AgentOpinion]
    consensus: bool  # all agents agree
    reasoning: str
    model_used: str


# Agent system prompts
AGENT_PROMPTS = {
    AgentRole.BULL: """You are a Bubble Detection Advocate analyzing LPPLS fit results.
Your role: argue WHY this IS a real phase transition / bubble.

Check these criteria:
- R² > 0.8 suggests good fit
- m ∈ (0.1, 0.9) is in valid range
- ω ∈ (6, 13) shows log-periodic oscillation
- B < 0 confirms super-exponential growth
- damping > 0.5 means power law dominates oscillation

Respond in JSON: {"verdict": "BUBBLE" or "NO_BUBBLE", "confidence": 0.0-1.0, "arguments": ["arg1", "arg2"]}""",
    AgentRole.BEAR: """You are a False Positive Detector analyzing LPPLS fit results.
Your role: challenge the prediction and find reasons it might be WRONG.

Check for:
- Overfitting (7 parameters on short data)
- Residual autocorrelation (DW < 1.5 is bad)
- m or ω at boundary values (optimizer stuck)
- Very small |B| (weak signal)
- High R² but poor out-of-sample performance

Respond in JSON: {"verdict": "BUBBLE" or "NO_BUBBLE", "confidence": 0.0-1.0, "arguments": ["arg1", "arg2"]}""",
    AgentRole.SKEPTIC: """You are a Methodological Critic analyzing LPPLS fit results.
Your role: question the methodology itself.

Check:
- Is the fitting window appropriate?
- Are bounds too tight or too loose?
- How sensitive is tc to parameter changes?
- Could a simpler model explain the data?
- Is the sample size sufficient?

Respond in JSON: {"verdict": "BUBBLE" or "NO_BUBBLE" or "UNCERTAIN", "confidence": 0.0-1.0, "arguments": ["arg1", "arg2"]}""",
}

ARBITER_PROMPT = """You are the Arbiter synthesizing three expert opinions on an LPPLS bubble prediction.

Given the Bull (advocate), Bear (challenger), and Skeptic (critic) opinions,
produce a final weighted verdict.

Rules:
- If 3/3 agree → high confidence in that direction
- If 2/3 agree → moderate confidence
- If all disagree → UNCERTAIN with low confidence
- Weight by each agent's confidence score

Respond in JSON: {"verdict": "BUBBLE" or "NO_BUBBLE" or "UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "synthesis explanation"}"""


def _format_lppls_context(
    params: LPPLSParams | None,
    r_squared: float,
    durbin_watson: float,
    n_observations: int,
) -> str:
    """Format LPPLS fit results as context for agents."""
    if params is None:
        return "No valid LPPLS fit found. All parameters failed Sornette filters."
    # WHY: raw values only — no directional annotations, no is_bubble verdict.
    # Each agent's system prompt already contains interpretation criteria.
    # Leaking is_bubble or hints like "(negative = bubble)" creates confirmation bias.
    return (
        f"LPPLS Fit Results:\n"
        f"  tc = {params.tc:.1f}\n"
        f"  m = {params.m:.4f}\n"
        f"  omega = {params.omega:.4f}\n"
        f"  B = {params.B:.6f}\n"
        f"  |B|/|C| = {params.damping:.2f}\n"
        f"  R_squared = {r_squared:.4f}\n"
        f"  Durbin_Watson = {durbin_watson:.3f}\n"
        f"  N_observations = {n_observations}"
    )


def _call_ollama(
    prompt: str,
    model: str = "qwen2.5:14b",
    temperature: float = 0.3,
    timeout: int = 60,
) -> str | None:
    """Call Ollama API for local inference."""
    try:
        import urllib.request

        data = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 300},
            }
        ).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")
    except Exception as e:
        log.warning("ollama_call_failed", error=str(e))
        return None


def _parse_agent_response(raw: str, role: AgentRole) -> AgentOpinion:
    """Parse JSON response from agent, with fallback."""
    try:
        # Find JSON in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            # WHY: LLM self-reported confidence is uncalibrated.
            # Clamp to [0.3, 0.8] to prevent any single agent from dominating.
            raw_conf = float(data.get("confidence", 0.5))
            clamped_conf = max(0.3, min(0.8, raw_conf))
            return AgentOpinion(
                role=role,
                verdict=data.get("verdict", "UNCERTAIN"),
                confidence=clamped_conf,
                reasoning=data.get("reasoning", ""),
                key_arguments=data.get("arguments", []),
            )
    except (json.JSONDecodeError, ValueError):
        pass

    # WHY: old fallback was BUBBLE-biased — "this is not a bubble" matched "bubble".
    # Now check negation patterns first, default to UNCERTAIN (not BUBBLE).
    raw_lower = raw.lower()
    if "no_bubble" in raw_lower or "not a bubble" in raw_lower or "no bubble" in raw_lower:
        verdict = "NO_BUBBLE"
    elif "uncertain" in raw_lower:
        verdict = "UNCERTAIN"
    elif "bubble" in raw_lower:
        verdict = "BUBBLE"
    else:
        verdict = "UNCERTAIN"

    return AgentOpinion(
        role=role,
        verdict=verdict,
        confidence=0.5,
        reasoning=raw[:200],
        key_arguments=[],
    )


def run_council(
    params: LPPLSParams | None,
    r_squared: float,
    durbin_watson: float,
    n_observations: int,
    model: str = "qwen2.5:14b",
) -> CouncilVerdict:
    """Run the full adversarial council on an LPPLS prediction.

    Args:
        params: Fitted LPPLS parameters
        r_squared: R² of the fit
        durbin_watson: DW statistic
        n_observations: Number of data points
        model: Ollama model name

    Returns:
        CouncilVerdict with final verdict and all opinions
    """
    context = _format_lppls_context(params, r_squared, durbin_watson, n_observations)
    opinions: list[AgentOpinion] = []

    # Run 3 agents
    for role in [AgentRole.BULL, AgentRole.BEAR, AgentRole.SKEPTIC]:
        prompt = f"{AGENT_PROMPTS[role]}\n\nData to analyze:\n{context}"
        raw = _call_ollama(prompt, model=model)

        if raw is None:
            # Fallback: heuristic opinion based on role
            opinion = _heuristic_opinion(role, params, r_squared, durbin_watson)
        else:
            opinion = _parse_agent_response(raw, role)

        opinions.append(opinion)
        log.info(
            "agent_opinion",
            role=role.value,
            verdict=opinion.verdict,
            confidence=round(opinion.confidence, 2),
        )

    # Run arbiter
    opinions_text = "\n".join(
        f"{o.role.value}: verdict={o.verdict}, confidence={o.confidence:.2f}, "
        f"arguments={o.key_arguments}"
        for o in opinions
    )
    arbiter_prompt = (
        f"{ARBITER_PROMPT}\n\nAgent opinions:\n{opinions_text}\n\nOriginal data:\n{context}"
    )
    arbiter_raw = _call_ollama(arbiter_prompt, model=model)

    if arbiter_raw:
        arbiter = _parse_agent_response(arbiter_raw, AgentRole.ARBITER)
        final_verdict = arbiter.verdict
        final_confidence = arbiter.confidence
        reasoning = arbiter.reasoning
    else:
        # Heuristic arbiter: majority vote weighted by confidence
        final_verdict, final_confidence, reasoning = _heuristic_arbiter(opinions)

    consensus = len(set(o.verdict for o in opinions)) == 1

    log.info(
        "council_verdict",
        verdict=final_verdict,
        confidence=round(final_confidence, 2),
        consensus=consensus,
    )

    return CouncilVerdict(
        final_verdict=final_verdict,
        final_confidence=final_confidence,
        opinions=opinions,
        consensus=consensus,
        reasoning=reasoning,
        model_used=model,
    )


def _heuristic_opinion(
    role: AgentRole,
    params: LPPLSParams | None,
    r_squared: float,
    durbin_watson: float,
) -> AgentOpinion:
    """Deterministic fallback when Ollama unavailable."""
    if params is None:
        return AgentOpinion(
            role=role,
            verdict="NO_BUBBLE",
            confidence=0.9,
            reasoning="No valid fit",
            key_arguments=["no_valid_fit"],
        )

    if role == AgentRole.BULL:
        score = 0.0
        args = []
        if r_squared > 0.8:
            score += 0.3
            args.append(f"R²={r_squared:.3f}>0.8")
        if 0.1 < params.m < 0.9:
            score += 0.2
            args.append(f"m={params.m:.3f} in range")
        if 6 < params.omega < 13:
            score += 0.2
            args.append(f"ω={params.omega:.2f} in range")
        if params.B < 0:
            score += 0.2
            args.append(f"B={params.B:.4f}<0")
        if params.damping > 0.5:
            score += 0.1
            args.append(f"damping={params.damping:.2f}>0.5")
        verdict = "BUBBLE" if score >= 0.5 else "NO_BUBBLE"
        return AgentOpinion(
            role=role,
            verdict=verdict,
            confidence=score,
            reasoning="Checklist scoring",
            key_arguments=args,
        )

    elif role == AgentRole.BEAR:
        score = 0.0
        args = []
        if durbin_watson < 1.5:
            score += 0.3
            args.append(f"DW={durbin_watson:.3f}<1.5 autocorrelation")
        if abs(params.B) < 0.01:
            score += 0.3
            args.append(f"|B|={abs(params.B):.4f} too small")
        if params.m > 0.85 or params.m < 0.15:
            score += 0.2
            args.append(f"m={params.m:.3f} near boundary")
        if params.omega > 12.5 or params.omega < 6.5:
            score += 0.2
            args.append(f"ω={params.omega:.2f} near boundary")
        verdict = "NO_BUBBLE" if score >= 0.5 else "BUBBLE"
        return AgentOpinion(
            role=role,
            verdict=verdict,
            confidence=max(score, 1 - score),
            reasoning="Red flag scoring",
            key_arguments=args,
        )

    else:  # SKEPTIC
        args = []
        if params.m > 0.85 or params.m < 0.15:
            args.append("m at boundary — optimizer may be stuck")
        if durbin_watson < 1.0:
            args.append("severe autocorrelation — model misspecified")
        if r_squared > 0.99:
            args.append("suspiciously high R² — possible overfit")
        verdict = "UNCERTAIN" if args else "BUBBLE"
        confidence = 0.4 if args else 0.6
        return AgentOpinion(
            role=role,
            verdict=verdict,
            confidence=confidence,
            reasoning="Methodology check",
            key_arguments=args,
        )


def _heuristic_arbiter(opinions: list[AgentOpinion]) -> tuple[str, float, str]:
    """Majority vote weighted by confidence."""
    votes: dict[str, float] = {}
    for o in opinions:
        votes[o.verdict] = votes.get(o.verdict, 0) + o.confidence

    best = max(votes, key=votes.get)
    total = sum(votes.values())
    confidence = votes[best] / total if total > 0 else 0.5

    return best, confidence, f"Weighted majority: {votes}"
