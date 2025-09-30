from agnt5.context import ContextVersion, get_context_version


def test_core_flag_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv("AGNT5_SDK_USE_CORE_CONTEXT", raising=False)
    assert get_context_version() is ContextVersion.LEGACY


def test_core_flag_enabled(monkeypatch):
    monkeypatch.setenv("AGNT5_SDK_USE_CORE_CONTEXT", "1")
    assert get_context_version() is ContextVersion.CORE
