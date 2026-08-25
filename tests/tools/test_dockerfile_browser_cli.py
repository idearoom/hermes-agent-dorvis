"""Container contract for the production Browser CLI installation."""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def test_browser_cli_engine_is_exactly_pinned_and_exposed_under_expected_name():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    version_match = re.search(
        r"^ARG BROWSER_HARNESS_VERSION=([^\s]+)$", dockerfile, re.MULTILINE
    )
    assert version_match, "Dockerfile must pin the Browser Harness version"
    version = version_match.group(1)
    assert re.search(
        r'"browser-harness>=\$\{BROWSER_HARNESS_VERSION\},<0\.3"',
        dockerfile,
    )
    assert "--resolution lowest-direct" in dockerfile
    assert "ln -sf /opt/hermes/bin/browser-harness /opt/hermes/bin/browser-use" in dockerfile
    assert (
        '/opt/hermes/bin/browser-use --version | grep -Fx "${BROWSER_HARNESS_VERSION}"'
        in dockerfile
    )
    assert version == "0.1.9"


def test_browser_cli_install_lives_outside_runtime_data_volume():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ENV UV_TOOL_DIR=/opt/hermes/.uv-tools" in dockerfile
    assert "ENV UV_TOOL_BIN_DIR=/opt/hermes/bin" in dockerfile
    assert "UV_TOOL_DIR=/opt/data" not in dockerfile
