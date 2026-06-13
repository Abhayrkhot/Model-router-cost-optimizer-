"""Unit tests for the model router simulator."""

import importlib
import os
import sys

import pytest

# Adjust path to import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa

from router import classify_prompt, route_prompt, simulate_requests, ModelTier


def test_classify_prompt():
    """The classifier should return expected categories based on token length."""
    assert classify_prompt("short prompt") == "easy"
    assert classify_prompt("word " * 25) == "medium"
    assert classify_prompt("word " * 60) == "hard"
    assert classify_prompt("") == "needs_info"


def test_route_prompt():
    """Routing should map classifications to the corresponding model tier."""
    models = {
        "small": ModelTier(name="small", cost_per_token=0.1, latency_ms_per_token=1.0),
        "medium": ModelTier(name="medium", cost_per_token=0.2, latency_ms_per_token=2.0),
        "large": ModelTier(name="large", cost_per_token=0.5, latency_ms_per_token=5.0),
    }
    # Should return small model for easy
    assert route_prompt("easy", models).name == "small"
    # Medium
    assert route_prompt("medium", models).name == "medium"
    # Hard
    assert route_prompt("hard", models).name == "large"
    # Unknown classification defaults to medium
    assert route_prompt("needs_info", models).name == "medium"


def test_simulation_cost_reduction():
    """Simulating a workload with routing should cost less than always using the largest model."""
    # Configure model costs so that large model is more expensive
    models = {
        "small": ModelTier(name="small", cost_per_token=0.001, latency_ms_per_token=10.0),
        "medium": ModelTier(name="medium", cost_per_token=0.002, latency_ms_per_token=20.0),
        "large": ModelTier(name="large", cost_per_token=0.01, latency_ms_per_token=50.0),
    }
    # Simulate 100 requests with 50% easy, 30% medium, 20% hard
    summary = simulate_requests(
        num_requests=100,
        easy_ratio=0.5,
        medium_ratio=0.3,
        seed=123,
        models=models,
    )
    # Compute baseline cost if every request used the large model
    # Approximate total tokens from the simulation summary
    baseline_total_cost = summary["total_tokens"] * models["large"].cost_per_token
    # Ensure the routed cost is less than or equal to baseline cost
    assert summary["total_cost"] <= baseline_total_cost
