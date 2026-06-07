"""Shared transcription configuration and provider resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TranscriptionProviderName = Literal["groq", "openai"]

_DEFAULT_PROVIDER: TranscriptionProviderName = "groq"
_DEFAULT_MODELS: dict[TranscriptionProviderName, str] = {
    "groq": "whisper-large-v3",
    "openai": "whisper-1",
}


@dataclass(frozen=True)
class EffectiveTranscriptionConfig:
    enabled: bool
    provider: TranscriptionProviderName
    model: str
    language: str | None
    api_key: str = field(repr=False)
    api_base: str
    max_duration_sec: int
    max_upload_mb: int

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def _as_provider(value: Any) -> TranscriptionProviderName | None:
    if isinstance(value, str):
        name = value.strip().lower()
        if name in _DEFAULT_MODELS:
            return name  # type: ignore[return-value]
    return None


def _provider_config(config: Any, provider: str) -> Any:
    return getattr(getattr(config, "providers", None), provider, None)


def resolve_transcription_config(config: Any) -> EffectiveTranscriptionConfig:
    """Resolve top-level transcription settings with legacy channel fallback."""
    top = getattr(config, "transcription", None)
    channels = getattr(config, "channels", None)
    provider = (
        _as_provider(getattr(top, "provider", None))
        or _as_provider(getattr(channels, "transcription_provider", None))
        or _DEFAULT_PROVIDER
    )
    provider_cfg = _provider_config(config, provider)
    return EffectiveTranscriptionConfig(
        enabled=bool(getattr(top, "enabled", True)),
        provider=provider,
        model=(getattr(top, "model", None) or _DEFAULT_MODELS[provider]).strip(),
        language=getattr(top, "language", None) or getattr(channels, "transcription_language", None),
        api_key=getattr(provider_cfg, "api_key", None) or "",
        api_base=getattr(provider_cfg, "api_base", None) or "",
        max_duration_sec=int(getattr(top, "max_duration_sec", 120)),
        max_upload_mb=int(getattr(top, "max_upload_mb", 25)),
    )


async def transcribe_audio_file(
    file_path: str | Path,
    config: EffectiveTranscriptionConfig,
) -> str:
    """Transcribe *file_path* using the already-resolved transcription config."""
    if not config.enabled or not config.configured:
        return ""
    if config.provider == "openai":
        from nanobot.providers.transcription import OpenAITranscriptionProvider

        provider = OpenAITranscriptionProvider(
            api_key=config.api_key,
            api_base=config.api_base or None,
            language=config.language,
            model=config.model,
        )
    else:
        from nanobot.providers.transcription import GroqTranscriptionProvider

        provider = GroqTranscriptionProvider(
            api_key=config.api_key,
            api_base=config.api_base or None,
            language=config.language,
            model=config.model,
        )
    return await provider.transcribe(file_path)
