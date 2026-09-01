# built-in dependencies
import tomllib
from pathlib import Path

# project dependencies
from face_engine import DeepFace
from face_engine.commons.logger import Logger

logger = Logger()


def test_version():
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    assert DeepFace.__version__ == pyproject["project"]["version"]
    logger.info("✅ versions are matching in both pyproject.toml and face_engine/__init__.py")
