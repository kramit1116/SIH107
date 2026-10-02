"""RAG configuration: env + config.yaml `rag`/`llm` sections. Backward compatible.

Switches:
  BIS_RAG_ENABLED=1|true|yes      enable lab retrieval in /chat (default on)
  BIS_RAG_DB_PATH=kb/bis_rag.db   SQLite corpus index
  BIS_RAG_TOP_K=5                 evidence chunks per query
  BIS_RAG_MINIMUM_RELEVANCE=0.25  minimum normalized lexical relevance [0,1]
  BIS_RAG_CATALOGUE_TOP_K=5       maximum catalogue candidates per query [1,100]
  BIS_RAG_SEMANTIC=1              enable dense retrieval (default on)
  BIS_RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
                                  model used when building/searching the dense index
  BIS_RAG_RERANKER_MODEL=""       optional sentence-transformers CrossEncoder
  BIS_LLM_MODEL=""                e.g. gpt-4o-mini / gemini-2.0-flash / llama3
  BIS_LLM_PROVIDER="openai-compatible"  openai-compatible | gemini | ollama | anthropic
  BIS_LLM_API_KEY=""              never hard-code; env/config only
  BIS_LLM_FALLBACK_API_KEYS=""     comma-separated backup keys, tried in order
  BIS_LLM_BASE_URL="https://api.openai.com/v1"  provider-specific local/cloud endpoint
  BIS_LLM_TEMPERATURE=0.2
  BIS_LLM_MAX_TOKENS=768
  BIS_LLM_TIMEOUT_S=10
  BIS_LLM_RETRIES=1
"""
from __future__ import annotations

import math
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = str(REPO_ROOT / "kb" / "bis_rag.db")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _bounded_float(name: str, configured, default: float,
                   minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, configured if configured is not None else default)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        value = default
    if not math.isfinite(value):
        value = default
    return min(max(value, minimum), maximum)


def _bounded_int(name: str, configured, default: int,
                 minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, configured if configured is not None else default)
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        value = default
    return min(max(value, minimum), maximum)


def _cfg_section(name: str) -> dict:
    try:
        from .config import load as load_config
        cfg = load_config()
        sec = cfg.get(name, {})
        return dict(sec) if isinstance(sec, dict) else {}
    except Exception:
        return {}


def load_rag_config() -> dict:
    file_cfg = _cfg_section("rag")
    return {
        "enabled": _bool_env("BIS_RAG_ENABLED", bool(file_cfg.get("enabled", True))),
        "db_path": os.environ.get("BIS_RAG_DB_PATH", str(file_cfg.get("db_path", DEFAULT_DB))),
        "top_k": _int_env("BIS_RAG_TOP_K", int(file_cfg.get("top_k", 5))),
        "minimum_relevance": _bounded_float(
            "BIS_RAG_MINIMUM_RELEVANCE", file_cfg.get("minimum_relevance"),
            0.25, 0.0, 1.0),
        "catalogue_top_k": _bounded_int(
            "BIS_RAG_CATALOGUE_TOP_K", file_cfg.get("catalogue_top_k"),
            5, 1, 100),
        "semantic": _bool_env("BIS_RAG_SEMANTIC", bool(file_cfg.get("semantic", True))),
        "embedding_model": os.environ.get(
            "BIS_RAG_EMBEDDING_MODEL",
            str(file_cfg.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)),
        "reranker_model": os.environ.get(
            "BIS_RAG_RERANKER_MODEL", str(file_cfg.get("reranker_model", ""))),
        "weight_lexical": _float_env(
            "BIS_RAG_WEIGHT_LEXICAL", float(file_cfg.get("weight_lexical", 1.0))),
        "weight_semantic": _float_env(
            "BIS_RAG_WEIGHT_SEMANTIC", float(file_cfg.get("weight_semantic", 0.3))),
        "exact_boost": _float_env(
            "BIS_RAG_EXACT_BOOST", float(file_cfg.get("exact_boost", 50.0))),
    }


def load_llm_config() -> dict:
    file_cfg = _cfg_section("llm")
    provider = os.environ.get(
        "BIS_LLM_PROVIDER", str(file_cfg.get("provider", "openai-compatible")))
    provider = provider.strip().lower()
    # Provider-aware endpoint defaults: an explicit BIS_LLM_BASE_URL (or a
    # non-default file value) always wins; otherwise gemini/ollama get their
    # own defaults instead of inheriting the OpenAI one from config.yaml.
    _OPENAI_DEFAULT = "https://api.openai.com/v1"
    _PROVIDER_DEFAULTS = {
        "gemini": "https://generativelanguage.googleapis.com",
        "ollama": "http://localhost:11434",
        "anthropic": "https://api.anthropic.com",
    }
    if "BIS_LLM_BASE_URL" in os.environ:
        base = os.environ["BIS_LLM_BASE_URL"]
    elif str(file_cfg.get("base_url", "")) not in ("", _OPENAI_DEFAULT) \
            or provider not in _PROVIDER_DEFAULTS:
        base = str(file_cfg.get("base_url", _OPENAI_DEFAULT))
    else:
        base = _PROVIDER_DEFAULTS[provider]
    primary_key = os.environ.get("BIS_LLM_API_KEY", str(file_cfg.get("api_key", ""))).strip()
    fallback_keys = os.environ.get("BIS_LLM_FALLBACK_API_KEYS", file_cfg.get("fallback_api_keys", []))
    if isinstance(fallback_keys, str):
        fallback_keys = fallback_keys.split(",")
    api_keys = list(dict.fromkeys(
        key.strip() for key in [primary_key, *fallback_keys] if key.strip()))
    return {
        "provider": provider,
        "model": os.environ.get("BIS_LLM_MODEL", str(file_cfg.get("model", ""))),
        "api_key": api_keys[0] if api_keys else "",
        "api_keys": api_keys,
        "base_url": base.rstrip("/"),
        "temperature": _float_env(
            "BIS_LLM_TEMPERATURE", float(file_cfg.get("temperature", 0.2))),
        "max_tokens": _int_env(
            "BIS_LLM_MAX_TOKENS", int(file_cfg.get("max_tokens", 768))),
        "timeout_s": _float_env(
            "BIS_LLM_TIMEOUT_S", float(file_cfg.get("timeout_s", 10.0))),
        "retries": _int_env(
            "BIS_LLM_RETRIES", int(file_cfg.get("retries", 0))),
        # Second model on the same provider, used when the primary is rate
        # limited or failing. Empty disables the fallback.
        "fallback_model": os.environ.get(
            "BIS_LLM_FALLBACK_MODEL", str(file_cfg.get("fallback_model", "") or "")),
        # Small, fast model for query rewriting and chat titles. Empty uses the
        # fallback model, then the primary model.
        "utility_model": os.environ.get(
            "BIS_LLM_UTILITY_MODEL", str(file_cfg.get("utility_model", "") or "")),
    }


def is_enabled() -> bool:
    return load_rag_config()["enabled"]
