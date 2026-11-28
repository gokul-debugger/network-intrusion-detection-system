import json
from pathlib import Path
import shlex
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_codespaces_python_matches_the_frozen_dashboard_release():
    config = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())
    manifest = json.loads((ROOT / "dashboard_assets/reference_v1/manifest.json").read_text())
    major_minor = ".".join(manifest["python"].split(".")[:2])
    assert f"1-{major_minor}-" in config["image"]


def test_codespaces_installs_the_pinned_dashboard_dependencies():
    config = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())
    command = config["updateContentCommand"]
    assert "< packages.txt &&" in command
    tokens = shlex.split(command)
    assert tokens[-7:] == ["python", "-m", "pip", "install", "--user", "-r", "app/requirements.txt"]


def test_codespaces_keeps_dashboard_request_protections_enabled():
    config = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())
    server = shlex.split(config["postAttachCommand"]["server"])
    assert server == ["python", "-m", "streamlit", "run", "app/modern_app.py", "--server.address", "0.0.0.0"]
    settings = tomllib.loads((ROOT / ".streamlit/config.toml").read_text())
    assert settings["server"]["enableCORS"] is True
    assert settings["server"]["enableXsrfProtection"] is True
