from types import SimpleNamespace

from agnt5.worker._core import Worker


def make_worker() -> Worker:
    worker = Worker.__new__(Worker)
    worker.service_name = "customer-service"
    worker.service_version = "1.0.0"
    return worker


def component(component_type: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        component_type=component_type,
        name=name,
        metadata={},
        config={},
    )


def test_print_startup_banner_prints_dashboard_url_from_env(monkeypatch, capsys):
    dashboard_url = (
        "https://app.agnt5.com/projects/6106a9b8-b2fa-4896-89d9-16bcceb20c72/components"
    )
    monkeypatch.setenv("AGNT5_DASHBOARD_URL", dashboard_url)

    make_worker()._print_startup_banner([component("workflow", "travel_booking_workflow")])

    output = capsys.readouterr().out
    assert f"Dashboard: {dashboard_url}" in output


def test_print_startup_banner_omits_dashboard_url_when_env_missing(monkeypatch, capsys):
    monkeypatch.delenv("AGNT5_DASHBOARD_URL", raising=False)

    make_worker()._print_startup_banner([component("workflow", "travel_booking_workflow")])

    output = capsys.readouterr().out
    assert "Dashboard:" not in output
