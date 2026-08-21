from pathlib import Path


PLUGIN = Path(__file__).resolve().parent.parent / "desktop" / "plugin.js"


def test_desktop_toggle_uses_project_scoped_endpoint():
    source = PLUGIN.read_text()

    assert "'/enabled?root=' + encodeURIComponent(cwd)" in source
    assert "Applies to this project only" in source
    assert "ctx.rest('/config', { method: 'PUT', body: { enabled: next } })" not in source
    assert "checked: !!live && live.enabled" in source


def test_settings_form_no_longer_owns_enabled_state():
    source = PLUGIN.read_text()

    assert "enabled: cfg.enabled !== false" not in source
    assert "checked: form.enabled" not in source
