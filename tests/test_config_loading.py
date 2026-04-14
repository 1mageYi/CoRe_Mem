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
    assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.benchmarks.personamem.variant == "32k"


def test_load_project_config_reads_gemini_trial_config():
    config = load_project_config(REPO_ROOT / "configs" / "gemini_flash.yaml")
    assert config.llm.provider == "google_ai_studio_openai_compatible"
    assert config.llm.api_key_env == "GEMINI_API_KEY"
    assert config.llm.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert config.llm.model == "gemini-2.5-flash"
    assert config.llm.timeout_seconds == 45
    assert config.llm.max_retries == 1
    assert config.llm.retry_backoff_seconds == 1.0
    assert config.llm.min_request_interval_seconds == 8.0
    assert config.llm.max_retry_delay_seconds == 5.0


def test_load_project_config_reads_gemini_ultraslow_config():
    config = load_project_config(REPO_ROOT / "configs" / "gemini_flash_ultraslow.yaml")
    assert config.llm.timeout_seconds == 30
    assert config.llm.max_retries == 0
    assert config.llm.min_request_interval_seconds == 30.0
