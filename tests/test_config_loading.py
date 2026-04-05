from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.config import load_project_config


def test_load_project_config_reads_stage1_defaults():
    config = load_project_config(REPO_ROOT / "configs" / "defaults.yaml")
    assert config.name == "core_mem"
    assert config.stage == "stage1"
    assert config.llm.api_key_env == "ALIYUN_API_KEY"
    assert config.benchmarks.personamem.variant == "32k"
