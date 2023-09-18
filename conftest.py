import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--yaml-path",
        action="store",
        default="default_yaml_path.yaml",
        help="Path to the YAML file",
    )


@pytest.fixture
def yaml_path(request):
    return request.config.getoption("--yaml-path")
