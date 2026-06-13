"""Model router simulation for cost–latency optimisation.

This module implements a simple LLM router that classifies prompts into
difficulty tiers and routes them to different model sizes accordingly.  A
command‑line interface allows you to simulate a workload with configurable
difficulty distribution and model configurations.  The simulator reports
aggregate cost and latency metrics.

The inspiration for this project comes from MorphLLM’s LLM router, which
classifies prompt difficulty and routes each request to the most cost‑efficient
model tier, achieving 40–70 % cost savings with minimal quality loss
【261533802441595†L39-L43】.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelTier:
    """Configuration for a model tier.

    Attributes
    ----------
    name: str
        Human‑readable name of the tier.
    cost_per_token: float
        Cost per generated token (arbitrary units).
    latency_ms_per_token: float
        Latency per generated token in milliseconds.
    """

    name: str
    cost_per_token: float
    latency_ms_per_token: float


def classify_prompt(prompt: str) -> str:
    """Classify a prompt into easy, medium or hard.

    This heuristic uses the number of whitespace‑separated tokens as a proxy for
    difficulty.  In production, a trained classifier would provide better
    accuracy【261533802441595†L145-L152】.

    Parameters
    ----------
    prompt: str
        The input prompt to classify.

    Returns
    -------
    str
        One of ``'easy'``, ``'medium'``, ``'hard'`` or ``'needs_info'``.
    """
    tokens = prompt.strip().split()
    n = len(tokens)
    if n == 0:
        # no tokens: we need more information from the user
        return "needs_info"
    if n < 20:
        return "easy"
    if n < 50:
        return "medium"
    return "hard"


def route_prompt(classification: str, models: Dict[str, ModelTier]) -> ModelTier:
    """Choose the appropriate model tier based on prompt classification.

    Parameters
    ----------
    classification: str
        Result from :func:`classify_prompt`.
    models: Dict[str, ModelTier]
        Mapping of tier names to model configurations.

    Returns
    -------
    ModelTier
        The selected model tier.  Unknown classifications default to the
        medium model.
    """
    if classification == "easy":
        return models["small"]
    if classification == "medium":
        return models["medium"]
    if classification == "hard":
        return models["large"]
    # needs_info or unknown
    return models["medium"]


def generate_random_prompt(category: str) -> str:
    """Generate a synthetic prompt string for simulation.

    Each category corresponds to a token length range.  The returned prompt
    contains dummy words separated by spaces.
    """
    if category == "easy":
        n = random.randint(5, 15)
    elif category == "medium":
        n = random.randint(20, 40)
    else:  # hard
        n = random.randint(50, 80)
    # Use the category name as the word to make prompts recognisable
    return (category + " ") * n


def simulate_requests(
    num_requests: int,
    easy_ratio: float,
    medium_ratio: float,
    seed: Optional[int] = None,
    models: Optional[Dict[str, ModelTier]] = None,
    ) -> Dict[str, object]:
    """Simulate a workload of prompts and route them to model tiers.

    Parameters
    ----------
    num_requests: int
        Total number of requests to simulate.
    easy_ratio: float
        Fraction of requests that should be ``easy``.
    medium_ratio: float
        Fraction of requests that should be ``medium``.  The remainder
        (1 - easy_ratio - medium_ratio) will be ``hard``.
    seed: Optional[int]
        Random seed for reproducibility.
    models: Optional[Dict[str, ModelTier]]
        Custom model tier configurations.  Defaults to a typical
        small/medium/large hierarchy.

    Returns
    -------
    Dict[str, object]
        A summary containing request counts, total and average cost and latency
        and per‑model counts.
    """
    rng = random.Random(seed)
    if models is None:
        models = {
            "small": ModelTier(name="small", cost_per_token=0.001, latency_ms_per_token=20.0),
            "medium": ModelTier(name="medium", cost_per_token=0.002, latency_ms_per_token=40.0),
            "large": ModelTier(name="large", cost_per_token=0.005, latency_ms_per_token=80.0),
        }
    categories = []
    for _ in range(num_requests):
        r = rng.random()
        if r < easy_ratio:
            categories.append("easy")
        elif r < easy_ratio + medium_ratio:
            categories.append("medium")
        else:
            categories.append("hard")
    total_tokens = 0
    total_cost = 0.0
    total_latency = 0.0
    model_counts = {m.name: 0 for m in models.values()}
    for category in categories:
        prompt = generate_random_prompt(category)
        classification = classify_prompt(prompt)
        model = route_prompt(classification, models)
        # Count tokens and accumulate cost and latency
        n_tokens = len(prompt.strip().split())
        total_tokens += n_tokens
        total_cost += n_tokens * model.cost_per_token
        total_latency += n_tokens * model.latency_ms_per_token
        model_counts[model.name] += 1
    summary = {
        "requests": num_requests,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "average_cost": total_cost / num_requests if num_requests > 0 else 0.0,
        "total_latency_ms": total_latency,
        "average_latency_ms": total_latency / num_requests if num_requests > 0 else 0.0,
        "model_counts": model_counts,
    }
    return summary


def cli(argv: Optional[List[str]] = None) -> None:
    """Command‑line interface for the model router simulator."""
    parser = argparse.ArgumentParser(description="Simulate an LLM router that routes prompts to model tiers based on difficulty.")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests to simulate")
    parser.add_argument("--easy-ratio", type=float, default=0.6, help="Fraction of requests classified as easy (0–1)")
    parser.add_argument("--medium-ratio", type=float, default=0.25, help="Fraction of requests classified as medium (0–1)")
    parser.add_argument("--small-cost", type=float, default=0.001, help="Cost per token for the small model")
    parser.add_argument("--medium-cost", type=float, default=0.002, help="Cost per token for the medium model")
    parser.add_argument("--large-cost", type=float, default=0.005, help="Cost per token for the large model")
    parser.add_argument("--small-latency", type=float, default=20.0, help="Latency in ms per token for the small model")
    parser.add_argument("--medium-latency", type=float, default=40.0, help="Latency in ms per token for the medium model")
    parser.add_argument("--large-latency", type=float, default=80.0, help="Latency in ms per token for the large model")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args(argv)
    # Create model tiers from CLI parameters
    models = {
        "small": ModelTier(name="small", cost_per_token=args.small_cost, latency_ms_per_token=args.small_latency),
        "medium": ModelTier(name="medium", cost_per_token=args.medium_cost, latency_ms_per_token=args.medium_latency),
        "large": ModelTier(name="large", cost_per_token=args.large_cost, latency_ms_per_token=args.large_latency),
    }
    result = simulate_requests(
        num_requests=args.num_requests,
        easy_ratio=args.easy_ratio,
        medium_ratio=args.medium_ratio,
        seed=args.seed,
        models=models,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()