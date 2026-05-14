from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Type

from core.env import get_env_class, list_registered_envs
from env.registry import (
    _import_android_gym,
    _import_dab_env,
    _import_deepeyes_env,
    _import_dw_env,
    _import_emb_env,
    _import_geo3k_vl_test_env,
    _import_math500_text_env,
    _import_mc_env,
    _import_mc_gpu_env,
    _import_os_env,
    _import_qa_gym,
    _import_robotrustbench_env,
)

EnvImporter = Callable[[], Type[Any]]

_ENV_IMPORTERS: Dict[str, EnvImporter] = {
    "android_gym": _import_android_gym,
    "mc": _import_mc_env,
    "mc_gym": _import_mc_env,
    "emb": _import_emb_env,
    "embodied_alfred": _import_emb_env,
    "os_gym": _import_os_env,
    "mc_gpu": _import_mc_gpu_env,
    "mc_gpu_gym": _import_mc_gpu_env,
    "geo3k_vl_test": _import_geo3k_vl_test_env,
    "math500_text": _import_math500_text_env,
    "math500": _import_math500_text_env,
    "qa_gym": _import_qa_gym,
    "deepeyes_env": _import_deepeyes_env,
    "dabstepgym": _import_dab_env,
    "discoveryworld": _import_dw_env,
    "robotrustbench": _import_robotrustbench_env,
}


def list_supported_env_names() -> List[str]:
    names = set(_ENV_IMPORTERS.keys())
    names.update(list_registered_envs().keys())
    return sorted(names)


def is_supported_env_name(env_name: str) -> bool:
    normalized = str(env_name or "").strip()
    if not normalized:
        return False
    if normalized in _ENV_IMPORTERS:
        return True
    return normalized in list_registered_envs()


def resolve_env_class(env_name: str) -> Type[Any]:
    normalized = str(env_name or "").strip()
    if not normalized:
        raise ValueError("env name is required")

    importer = _ENV_IMPORTERS.get(normalized)
    if importer is not None:
        return importer()

    try:
        return get_env_class(normalized)
    except Exception as exc:
        available = ", ".join(list_supported_env_names())
        raise ValueError(f"Unknown envname: {normalized} (available: {available})") from exc


def normalize_create_kwargs(raw_value: Any) -> Dict[str, Any]:
    """Normalize env creation kwargs from DB rows or HTTP payloads."""
    if raw_value is None:
        return {}

    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid env_param JSON string: {exc}") from exc
        if not isinstance(parsed, dict):
            raise TypeError("env_param JSON string must decode to an object.")
        return dict(parsed)

    if isinstance(raw_value, dict):
        return dict(raw_value)

    raise TypeError("env_param must be a JSON object or JSON string.")
