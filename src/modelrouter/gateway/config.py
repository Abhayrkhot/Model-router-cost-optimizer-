"""Gateway configuration (YAML file + environment overrides).

The router artifact predicts quality for the models it was trained on.  Each of those names must
map to a *deployment*: a concrete upstream (provider type, upstream model id, prices, timeouts).
Deployments can also be called directly by name, bypassing the router.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Deployment(BaseModel):
    name: str
    provider: Literal["anthropic", "openai_compat", "replay"]
    upstream_model: str
    base_url: str | None = None
    api_key_env: str | None = None
    # USD per million tokens; replay deployments report the recorded cost instead.
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    timeout_s: float = 60.0
    max_retries: int = 1
    max_tokens_default: int = 1024
    # Anthropic only: let the API re-run a refused request on a fallback model (Opus/Sonnet 5.5,
    # Fable 5.2).  The gateway also falls back across deployments on refusals either way.
    server_side_fallback: bool = True
    # Replay-only knobs (latency model and fault injection for tests/benchmarks).
    latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    failure_rate: float = 0.0


class Tenant(BaseModel):
    name: str
    # sha256 hex of the bearer key; plaintext keys never live in config.
    key_sha256: str
    daily_budget_usd: float = 10.0
    requests_per_minute: int = 600
    default_lambda: float = 300.0

    @staticmethod
    def hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()


class BreakerConfig(BaseModel):
    failure_threshold: int = 5  # consecutive failures that open the breaker
    reset_after_s: float = 10.0  # open -> half-open
    half_open_probes: int = 1


class GatewayConfig(BaseModel):
    router_dir: Path
    # router model name -> deployment name
    model_map: dict[str, str]
    deployments: list[Deployment]
    tenants: list[Tenant]
    redis_url: str | None = None
    request_deadline_s: float = 90.0
    max_attempts: int = 3  # deployments tried per request (primary + fallbacks)
    cache_ttl_s: int = 0  # 0 disables the exact-match response cache
    decision_log: Path | None = None
    replay_fixture: Path | None = None  # required when any deployment uses provider: replay
    # Exposes POST /admin/faults to change replay failure rates at runtime (benchmarks only).
    fault_injection: bool = False
    embed_threads: int = 2
    batch_max_size: int = 32
    batch_max_wait_ms: float = 2.0
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)

    @model_validator(mode="after")
    def _check_refs(self) -> GatewayConfig:
        names = {d.name for d in self.deployments}
        if len(names) != len(self.deployments):
            raise ValueError("deployment names must be unique")
        missing = sorted(set(self.model_map.values()) - names)
        if missing:
            raise ValueError(f"model_map points at unknown deployments: {missing}")
        return self

    def deployment(self, name: str) -> Deployment:
        for d in self.deployments:
            if d.name == name:
                return d
        raise KeyError(name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GatewayConfig:
        raw = yaml.safe_load(Path(path).read_text())
        base = Path(path).resolve().parent
        cfg = cls.model_validate(raw)
        if not cfg.router_dir.is_absolute():
            cfg.router_dir = (base / cfg.router_dir).resolve()
        if cfg.decision_log is not None and not cfg.decision_log.is_absolute():
            cfg.decision_log = (base / cfg.decision_log).resolve()
        if cfg.replay_fixture is not None and not cfg.replay_fixture.is_absolute():
            cfg.replay_fixture = (base / cfg.replay_fixture).resolve()
        return cfg


class Settings(BaseSettings):
    """Process settings: where the config lives, plus env overrides for secrets/infra."""

    model_config = SettingsConfigDict(env_prefix="MODELROUTER_")

    config: Path = Path("deploy/gateway.yaml")
    redis_url: str | None = None
    log_level: str = "info"
