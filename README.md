# Model Router: Cost–Latency Optimizer

## Overview

LLM inference is not one‑size‑fits‑all.  Different tasks require different
levels of reasoning.  Using a single high‑end model for every request can
dramatically overpay for simple tasks.  An **LLM router** examines the
incoming prompt and routes it to the most appropriate model tier (small,
medium or large) based on its difficulty.  MorphLLM’s blog explains that a
router classifies prompt difficulty and maps it to a model tier, achieving
40–70 % cost savings with less than 2 % quality loss and adding only ~430 ms of
latency【261533802441595†L39-L43】.  Easy prompts like adding a TODO comment or
renaming variables are served by a small, cheap model, while complex prompts
are sent to a large model【261533802441595†L115-L129】.

This project implements a simple **model router simulator**.  It classifies
prompts based on length (as a proxy for complexity), routes them to one of
three model tiers (small, medium, large), and estimates cost and latency.
A command‑line interface allows you to simulate workloads with different
difficulty distributions and measure the trade‑offs between cost and latency.

## Why this matters

Without routing, every request is sent to the most expensive model.  MorphLLM
notes that many coding prompts are simple boilerplate edits and can be handled
by a low‑cost model【261533802441595†L75-L90】.  Classifying prompts and routing
them appropriately can reduce costs by up to 70 % while maintaining quality
【261533802441595†L39-L43】.  This project demonstrates how routing affects
average cost and latency, making it an important technique for building
efficient LLM applications.

## Features

* **Prompt classification:** A heuristic classifier categorises prompts into
  easy, medium or hard based on token length.  In production, this would be
  replaced by a trained model【261533802441595†L145-L152】.
* **Model tier configuration:** Each model tier is described by a cost per
  token and latency per token.  Default configurations approximate a
  small/medium/large model hierarchy.
* **Routing:** Prompts are routed to the cheapest model capable of handling
  their complexity.  Easy prompts go to the small model; medium to medium; hard
  to the large model.  Prompts classified as “needs info” would normally
  prompt the user for clarification, but the simulator assigns them to the
  medium tier.
* **Simulation and metrics:** Simulate a stream of random prompts with a
  configurable difficulty distribution.  The simulator reports total cost,
  average cost per request, total latency and average latency.

## Installation

This project requires Python 3.8+ and no external dependencies apart from
[`pytest`](https://docs.pytest.org/) for the test suite.  You can install the
development dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulator from the command line.  The following example simulates
200 requests with a difficulty distribution of 60 % easy, 25 % medium and
15 % hard, using the default model configurations:

```bash
python router.py --num-requests 200 --easy-ratio 0.6 --medium-ratio 0.25 --seed 42
```

Sample output:

```
{
  "requests": 200,
  "total_tokens": 2314,
  "total_cost": 0.25253,
  "average_cost": 0.00126,
  "total_latency_ms": 12350.0,
  "average_latency_ms": 61.75,
  "model_counts": {
    "small": 122,
    "medium": 50,
    "large": 28
  }
}
```

The `model_counts` section shows how many prompts were routed to each model.

## Running the tests

To run the unit tests, execute:

```bash
pytest
```

The tests cover prompt classification, routing and simulation accuracy.

## Limitations and future work

This simulator uses a trivial classifier based solely on prompt length.  Real
routers use trained models to predict complexity【261533802441595†L145-L152】.
The cost and latency models are simplified and do not account for overheads
such as classification time (~430 ms per request)【261533802441595†L39-L43】.
Extending the simulator to include multi‑tier fallback and user feedback would
make it more realistic.

## Resume bullet

> Designed a model router simulator that classifies prompt difficulty, routes
> requests to small/medium/large model tiers, and estimates cost and latency
> trade‑offs, demonstrating how routing can achieve 40–70 % cost savings with
> minimal quality impact【261533802441595†L39-L43】.
