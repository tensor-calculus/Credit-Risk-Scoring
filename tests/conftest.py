import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root and api to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "api"))

from api.main import app

@pytest.fixture
def client():
    return TestClient(app)
