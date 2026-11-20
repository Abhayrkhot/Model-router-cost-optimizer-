# Model Router: Cost–Quality Optimizer

> **Status: being rebuilt.** The earlier version was a length-based simulator that never called
> a model and quoted savings figures from a blog post. This rewrite trains and evaluates routers
> on real model outcomes; results land here once they are measured.

An LLM router decides, per request, which model should answer it. A cheap model answers most
prompts as well as an expensive one, so routing the easy ones away from the expensive model cuts
cost. The open question is how much quality that costs, and only measured outcomes can answer it.

This project:

1. **Data.** Uses RouterBench: ~36.5k prompts from MMLU, HellaSwag, GSM8K, ARC, Winogrande,
   MBPP and MT-Bench, each answered by the same 11 models, with correctness and cost for every
   answer. GSM8K is re-graded by exact match against the official answers.
2. **Routers.** Learns which models will answer each prompt correctly, then picks the model
   with the best expected quality per dollar.
3. **Evaluation.** Measures the full cost-vs-quality curve on held-out prompts and on held-out
   benchmarks, against single models, random mixing and an oracle.
4. **Gateway.** Serves the router behind an OpenAI-compatible API with fallbacks, budgets,
   caching and metrics.

## Development

Work runs in a container (`scripts/dev.sh`) with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run modelrouter data prepare   # download pinned RouterBench + GSM8K, build data/processed
uv run pytest
```
