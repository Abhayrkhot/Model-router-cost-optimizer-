# Design notes

## The routing decision

A router answers: *for this prompt, which model gives the best answer per dollar?* It's split into
three parts that can be tested and replaced separately.

1. **Quality model**: `q̂[i, m] ≈ P(model m answers prompt i well)` for every model in the pool.
   Training data is RouterBench: every prompt was answered by every model, so each (prompt,
   model) pair has a quality label and a cost.
2. **Cost model**: `ĉ[i, m]`, the predicted dollar cost of model m's answer. A per-model linear
   fit on prompt length (input tokens dominate on most of these benchmarks; output length
   varies by model and is absorbed in the intercept).
3. **Policy**: `choose argmax_m q̂[i, m] − λ · ĉ[i, m]`. λ is the price of quality in dollars:
   λ = 0 buys the best predicted answer at any price, large λ buys the cheapest. One λ is one
   operating point, so sweeping λ traces the router's cost–quality curve.

Why a per-model quality model instead of a classifier of "difficulty tiers"? Difficulty isn't one
number. Yi-34B is far better than Claude 2 on HellaSwag and worse on MBPP, and GPT-3.5 is cheaper
than Claude Instant and better at code. Predicting each model's success lets the policy use those
crossovers; a tier label can't.

Why λ instead of a confidence threshold? λ has a meaning operators can set ("an extra point of
quality is worth $x per 1k prompts"), it works for any number of models, and it allows a
deployment to *calibrate* it: choose the λ whose validation quality matches a target, then
serve with it. The calibrated numbers in [RESULTS.md](RESULTS.md) are exactly that procedure.

### Quality models tried

| Model | Input | Notes |
|---|---|---|
| TF-IDF ridge | words/bigrams | no embedding model; strong lexical baseline |
| kNN | bge-small embedding | k and shrinkage toward each model's prior tuned on val |
| Logistic / MLP | embedding + log length | soft-label BCE, early stopping on val; NumPy forward at serve time |
| kNN + MLP | both | averaged predictions; best AIQ and calibrated cost on every seed |

All of them land within ~1 AIQ point of each other on random splits. The gap to the oracle is
large (0.71 vs 0.92 AIQ), but most of it is not learnable: on multiple-choice benchmarks a weak
model is right by chance on a quarter of the prompts, and no router can know which quarter.

### Embeddings

`BAAI/bge-small-en-v1.5` through fastembed (quantized ONNX, CPU, no torch at serve time), capped
at 256 tokens. Prompts are embedded from the **router text**: the head of the first turn plus
the last turn. GSM8K prompts start with an identical few-shot block and the real question comes
last, so embedding the full prompt with head truncation would make every GSM8K prompt look the
same. Batches are length-sorted so short prompts aren't padded to long ones.

## Evaluation

* Labels and costs are realised outcomes: the score and cost of the answer the chosen model
  actually gave. Nothing is simulated.
* **GSM8K is re-graded.** RouterBench's `grade-school-math` column is a 5-level grade, and GPT-4's
  correct answers mostly score 0.75. All 7,450 × 11 responses were re-graded by exact match
  against the official answers. Last-number extraction (lm-eval's flexible rule) mis-grades chat
  models that restate the givens at the end ("6 bottles … for 180 days"), so the extractor reads
  the concluding sentence; see `data/gsm8k.py`. A hand audit of 36 random responses found no
  extraction errors.
* Splits: stratified random 70/10/20 (three seeds) and leave-one-benchmark-out.
* Baselines a router must beat: the best mix of single models (RouterBench's "zero router": the
  upper hull of the single-model points), the original length rule, random routing (two-model),
  and, from above, the oracle.
* Metrics: AIQ (area under the hull, linear and log cost), quality at fixed budgets, cost to reach
  90/95/98/100% of GPT-4's quality, and RouteLLM's APGR/CPT for GPT-4 vs Mixtral. Bootstrap CIs
  hold the policy fixed and resample test prompts.

## Gateway

```
client ─▶ auth ─▶ rate limit ─▶ cache? ─▶ router (micro-batch: embed + q̂ + ĉ) ─▶ ranked candidates
                                                                                    │
         ◀── response + routing info ◀── settle cost ◀── call w/ retries ◀── breaker? budget? ┘
                                                              │ fallback-eligible error
                                                              └──▶ next candidate
```

* **Ranked fallback.** The router returns every model in utility order, not one answer. If the
  first choice is overloaded, rate-limited, timing out, 5xx-ing or refuses, the gateway tries the
  next one. Client errors (a malformed request) are returned instead: they'd fail everywhere.
* **Circuit breakers** per deployment (closed → open after N consecutive failures → half-open
  probe after a cooldown). An open deployment is skipped during ranking, so traffic moves to the
  next-best model immediately instead of paying a timeout per request.
* **Budgets.** Reserve the predicted cost before the call, settle to the actual cost after. That
  stops a burst of concurrent requests from overshooting a tenant's daily budget together. When
  the top choice doesn't fit the remaining budget, a cheaper candidate is used; only when none
  fits does the request fail with 402. Redis Lua scripts make reserve and the token-bucket rate
  limit atomic across replicas.
* **Micro-batching.** Concurrent requests' embeddings are computed together (up to 32, or 2 ms)
  in a worker thread, so the event loop never blocks on ONNX.
* **Streaming** can fall back only until the first byte is sent; after that a failure is
  reported in-stream.
* **Replay provider.** Serves the response each RouterBench model really gave, with injected
  latency and failures. The whole HTTP path can be benchmarked and fault-tested for $0, and its
  quality and cost compared directly with the offline evaluation.
* **Claude provider** uses the official Anthropic SDK with its retries off (the gateway owns
  retries), does not forward sampling parameters (current API versions reject them), and treats
  a `refusal` stop reason as fallback-eligible. Deployments of Opus 5.5 / Sonnet 5.5 also opt into
  the API's server-side refusal fallback.
* **Decision log.** Every request's ranking, choice, attempts and cost go to a JSONL log off the
  request path. Joined with quality feedback, that is the training set for the next router.

## What this does not do

* The shipped router knows RouterBench's 11 (2023-era) models. Routing between other models
  needs outcomes for them; the data format and `modelrouter train` are model-agnostic.
* Quality is benchmark correctness. Production quality (helpfulness, user satisfaction) needs
  its own labels, such as judge scores or feedback, in the same format.
* Unseen task types generalise poorly (see the leave-one-benchmark-out table). A deployment
  should log decisions, label a sample, and retrain.
