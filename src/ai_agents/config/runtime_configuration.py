from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

from ai_agents.config.settings import settings
from ai_agents.config.constants import (
    ChatProvider,
    PROVIDER_CAPABILITIES,
    PUBLIC_FIELDS,
    PROVIDER_FIELDS,
    PROVIDER_URL_FIELDS,
    PROVIDER_SLOT_CAPABILITY
)




class RuntimeAgentConfigurationStore:
    """Persist non-secret model choices and apply session-only secret overrides.

    The renderer never receives secret values. Durable credentials should remain in
    environment variables or the existing Secrets Manager integration. This store
    only keeps credentials in the backend process for the current session.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        configured_path = path or settings.runtime_agent_config_path
        self.path = Path(configured_path).expanduser().resolve()
        self._lock = threading.RLock()
        self._load_and_apply()

    def _defaults(self) -> dict[str, Any]:
        return {field: getattr(settings, field) for field in PUBLIC_FIELDS}

    def _load_file(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        return raw if isinstance(raw, dict) else {}

    def _validate_public_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        result = self._defaults()

        for field in PUBLIC_FIELDS:
            if field not in values:
                continue
            value = values[field]

            if field in PROVIDER_SLOT_CAPABILITY:
                if value not in PROVIDER_FIELDS:
                    raise ValueError(f"Unsupported provider for {field}: {value}")
                result[field] = value
                continue

            if field == "voice_tts_enabled":
                if not isinstance(value, bool):
                    raise ValueError("voice_tts_enabled must be a boolean.")
                result[field] = value
                continue

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string.")

            result[field] = value.strip()

        for provider_field, capability in PROVIDER_SLOT_CAPABILITY.items():
            provider = result[provider_field]
            if capability not in PROVIDER_CAPABILITIES[provider]:
                raise ValueError(
                    f"Provider '{provider}' does not support the {capability} capability "
                    f"required by {provider_field}."
                )

        return result

    def _apply_public_values(self, values: Mapping[str, Any]) -> None:
        for field in PUBLIC_FIELDS:
            if field in values:
                setattr(settings, field, values[field])

    def _load_and_apply(self) -> None:
        with self._lock:
            values = self._validate_public_values(self._load_file())
            self._apply_public_values(values)

    def _write_public_values(self, values: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({field: values[field] for field in PUBLIC_FIELDS}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, self.path)

    def _apply_secrets(self, secrets: Mapping[str, Any] | None) -> None:
        if not secrets:
            return

        for provider, raw_value in secrets.items():
            if provider not in PROVIDER_FIELDS:
                raise ValueError(f"Unsupported secret provider: {provider}")
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue

            settings_field, environment_name, _ = PROVIDER_FIELDS[provider]  # type: ignore[index]
            value = raw_value.strip()
            setattr(settings, settings_field, value)
            os.environ[environment_name] = value

    def update(
        self,
        values: Mapping[str, Any],
        *,
        secrets: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            validated = self._validate_public_values(values)
            self._write_public_values(validated)
            self._apply_public_values(validated)
            self._apply_secrets(secrets)
            return self.public_snapshot()

    def secret_configured(self, provider: ChatProvider) -> bool:
        settings_field, environment_name, secret_arn_field = PROVIDER_FIELDS[provider]
        return bool(
            getattr(settings, settings_field, None)
            or os.getenv(environment_name)
            or (secret_arn_field and getattr(settings, secret_arn_field, None))
        )

    def provider_api_key(self, provider: ChatProvider) -> str | None:
        resolver_name = f"resolved_{provider}_api_key"
        resolver = getattr(settings, resolver_name, None)
        if callable(resolver):
            return resolver()

        settings_field, environment_name, _ = PROVIDER_FIELDS[provider]
        return getattr(settings, settings_field, None) or os.getenv(environment_name)

    def provider_base_url(self, provider: ChatProvider) -> str:
        value = getattr(settings, PROVIDER_URL_FIELDS[provider])
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"No base URL is configured for provider '{provider}'.")
        return value.rstrip("/")

    def public_snapshot(self) -> dict[str, Any]:
        snapshot = {field: getattr(settings, field) for field in PUBLIC_FIELDS}
        snapshot["secrets_configured"] = {
            provider: self.secret_configured(provider)
            for provider in PROVIDER_FIELDS
        }
        snapshot["secrets_persistence"] = "session_only"
        return snapshot


runtime_agent_configuration = RuntimeAgentConfigurationStore()
