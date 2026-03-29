import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_ingestion_module(project_root: Path):
    pipeline_code_dir = project_root / "ResPlan_RAG" / "pipeline_code"

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(pipeline_code_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_code_dir))

    pkg_config = importlib.import_module("ResPlan_RAG.pipeline_code.config")
    sys.modules["config"] = pkg_config

    try:
        return importlib.import_module("ResPlan_RAG.pipeline_code.data_ingestion")
    except ModuleNotFoundError as exc:
        pytest.skip(f"data_ingestion module unavailable in this workspace: {exc}")


@pytest.fixture(scope="session")
def retrieval_helpers_module(project_root: Path):
    pipeline_code_dir = project_root / "ResPlan_RAG" / "pipeline_code"

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(pipeline_code_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_code_dir))

    pkg_config = importlib.import_module("ResPlan_RAG.pipeline_code.config")
    sys.modules["config"] = pkg_config

    return importlib.import_module("ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine_helpers")


@pytest.fixture(scope="session")
def retrieval_engine_module(project_root: Path):
    pipeline_code_dir = project_root / "ResPlan_RAG" / "pipeline_code"

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(pipeline_code_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_code_dir))

    pkg_config = importlib.import_module("ResPlan_RAG.pipeline_code.config")
    sys.modules["config"] = pkg_config

    return importlib.import_module("ResPlan_RAG.pipeline_code.retrieval_engine.retrieval_engine")


